#!/usr/bin/env python3
"""
Complete Phase 2 Parameter Estimation Refinement System
Fixed version with proper baseline comparison and directory structure
"""

import numpy as np
import bilby
from bilby.gw.detector import PowerSpectralDensity, InterferometerList
from bilby.core.prior import Uniform, Sine, PriorDict
import matplotlib.pyplot as plt
import pickle
import os
from scipy.interpolate import interp1d
import logging
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime
import time
import corner
from gwpy.frequencyseries import FrequencySeries

# ============================================================================
# SECTION 1: ASD EVOLUTION MODEL
# ============================================================================

class ASD_Evolution_Model:
    """Models the evolution of ASD from preliminary to final calibration."""
    
    def __init__(self, base_asd_file: str, evolution_stages: int = 5):
        """
        Initialize ASD evolution model.
        
        Args:
            base_asd_file: Path to the final (true) ASD file
            evolution_stages: Number of calibration stages to simulate
        """
        self.base_asd_file = base_asd_file
        self.evolution_stages = evolution_stages
        self.load_base_asd()
        
    def load_base_asd(self):
        """Load the base (final) ASD from file."""
        with open(self.base_asd_file, 'rb') as f:
            data = pickle.load(f)
        self.base_asd = data['asd']
        self.frequencies = self.base_asd.frequencies.value
        self.base_values = self.base_asd.value
        
    def Generate_Evolved_ASD(self, stage: int) -> FrequencySeries:
        """
        Generate an evolved ASD for a given calibration stage.
        
        Args:
            stage: Calibration stage (0 = preliminary, evolution_stages-1 = final)
            
        Returns:
            Evolved ASD as FrequencySeries
        """
        # Calculate evolution factor (1.0 = final calibration, >1.0 = preliminary)
        evolution_factor = 1.0 + (self.evolution_stages - stage - 1) * 0.1
        
        # Add frequency-dependent systematic bias
        freq_bias = 1.0 + 0.05 * np.sin(2 * np.pi * np.log10(self.frequencies / 100))
        freq_bias *= (self.evolution_stages - stage - 1) / self.evolution_stages
        
        # Add statistical fluctuations
        noise_level = 0.05 * (self.evolution_stages - stage - 1) / self.evolution_stages
        statistical_noise = 1.0 + np.random.normal(0, noise_level, len(self.frequencies))
        
        # Combine effects
        evolved_values = self.base_values * evolution_factor * (1 + freq_bias) * statistical_noise
        
        # Create new FrequencySeries
        evolved_asd = FrequencySeries(
            evolved_values,
            frequencies=self.frequencies,
            unit=self.base_asd.unit
        )
        
        return evolved_asd
        
    def Plot_Evolution(self, save_path: str = None):
        """Plot the ASD evolution across calibration stages."""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, self.evolution_stages))
        
        for stage in range(self.evolution_stages):
            evolved_asd = self.Generate_Evolved_ASD(stage)
            label = f"Stage {stage}" if stage < self.evolution_stages - 1 else "Final"
            plt.loglog(self.frequencies, evolved_asd.value, 
                      color=colors[stage], label=label, alpha=0.8)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('ASD (strain/√Hz)')
        plt.title('ASD Evolution from Preliminary to Final Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(10, 1000)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# SECTION 2: PE REFINEMENT ENGINE
# ============================================================================

class PE_Refinement_Engine:
    """Handles parameter estimation refinement using evolved priors."""
    
    def __init__(self, outdir: str = 'refinement_results'):
        """
        Initialize PE refinement engine.
        
        Args:
            outdir: Output directory for results
        """
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_file = os.path.join(self.outdir, 'refinement.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def Create_Prior_From_Posterior(self, posterior_result: bilby.result.Result, 
                                   scale_factor: float = 1.5) -> PriorDict:
        """
        Create informed priors from posterior distributions.
        
        Args:
            posterior_result: Bilby result object from previous PE
            scale_factor: Factor to expand credible intervals for robustness
            
        Returns:
            Prior dictionary based on posterior
        """
        informed_priors = PriorDict()
        
        # Parameters to create informed priors for
        params_to_update = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 
                           'theta_jn', 'phase', 'geocent_time']
        
        for param in params_to_update:
            if param in posterior_result.posterior:
                samples = posterior_result.posterior[param].values
                
                # Calculate credible intervals
                lower, median, upper = np.percentile(samples, [5, 50, 95])
                width = (upper - lower) * scale_factor / 2
                
                # Create appropriate prior based on parameter type
                if param == 'theta_jn':
                    # Ensure bounds are within [0, pi]
                    informed_priors[param] = Sine(
                        minimum=max(0, median - width),
                        maximum=min(np.pi, median + width)
                    )
                elif param == 'phase':
                    # Handle phase wrapping
                    informed_priors[param] = Uniform(
                        minimum=max(0, median - width),
                        maximum=min(2*np.pi, median + width)
                    )
                else:
                    # Standard uniform prior
                    informed_priors[param] = Uniform(
                        minimum=median - width,
                        maximum=median + width
                    )
                    
                logging.info(f"Created informed prior for {param}: "
                           f"[{informed_priors[param].minimum:.3f}, "
                           f"{informed_priors[param].maximum:.3f}]")
        
        return informed_priors
        
    def Run_Refined_PE(self, interferometers: InterferometerList,
                      waveform_generator: bilby.gw.WaveformGenerator,
                      informed_priors: PriorDict,
                      injection_parameters: Dict,
                      label: str,
                      sampler_kwargs: Dict = None) -> bilby.result.Result:
        """
        Run refined parameter estimation with informed priors.
        """
        # Set up likelihood
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=interferometers,
            waveform_generator=waveform_generator
        )
        
        # Default sampler settings optimized for refinement
        default_sampler_kwargs = {
            'npoints': 300,  # Fewer points needed with informed priors
            'walks': 15,     # Fewer walks needed
            'dlogz': 0.1,
            'sample': 'rwalk',
            'bound': 'multi'
        }
        
        if sampler_kwargs:
            default_sampler_kwargs.update(sampler_kwargs)
            
        # Run sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=informed_priors,
            sampler='dynesty',
            outdir=self.outdir,
            label=label,
            injection_parameters=injection_parameters,
            save=True,
            **default_sampler_kwargs
        )
        
        return result


# ============================================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================================

def Create_Test_Injection_Parameters() -> Dict:
    """Create test injection parameters for validation."""
    return {
        'chirp_mass': 30.0,
        'mass_ratio': 0.9,
        'luminosity_distance': 450.0,
        'a_1': 0.0,
        'a_2': 0.0,
        'tilt_1': 0.0,
        'tilt_2': 0.0,
        'phi_12': 0.0,
        'phi_jl': 0.0,
        'theta_jn': 0.8,
        'phase': 1.0,
        'ra': 1.5,
        'dec': -1.0,
        'psi': 2.5,
        'geocent_time': 1238303719.0
    }


def Setup_Interferometers_With_ASD(asd_data: Dict, injection_parameters: Dict,
                                  duration: float = 4, sampling_frequency: float = 2048,
                                  minimum_frequency: float = 20) -> Tuple:
    """
    Set up interferometers with specific ASD.
    """
    # Create interferometer list
    interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    
    # Calculate frequency array
    n_freq = int(duration * sampling_frequency / 2) + 1
    frequencies = np.linspace(0, sampling_frequency/2, n_freq)
    
    # Set up waveform generator for injection
    waveform_arguments = dict(
        waveform_approximant='IMRPhenomD',
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency
    )
    
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments
    )
    
    # Configure each interferometer
    geocent_time = injection_parameters['geocent_time']
    start_time = geocent_time - duration + 0.5
    
    for ifo in interferometers:
        # Get ASD for this detector
        asd = asd_data[ifo.name]
        
        # Interpolate ASD to match frequency array
        asd_interpolator = interp1d(
            asd.frequencies.value, 
            asd.value,
            bounds_error=False, 
            fill_value='extrapolate'
        )
        interpolated_asd = asd_interpolator(frequencies)
        
        # Set interferometer properties
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = sampling_frequency/2
        ifo.sampling_frequency = sampling_frequency
        ifo.duration = duration
        ifo.start_time = start_time
        
        # Set PSD from ASD
        psd_array = interpolated_asd ** 2
        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=frequencies,
            psd_array=psd_array
        )
        
        # Initialize strain data
        ifo.strain_data.roll_off = 0.2
        ifo.strain_data.set_from_frequency_domain_strain(
            sampling_frequency=sampling_frequency,
            duration=duration,
            frequency_domain_strain=np.zeros(n_freq, dtype=complex)
        )
        
        # Inject signal
        ifo.inject_signal(
            parameters=injection_parameters,
            waveform_generator=waveform_generator
        )
        
        logging.info(f"Set up {ifo.name} with evolved ASD and injected signal")
        
    return interferometers, waveform_generator


def Run_Phase1_PE(interferometers: bilby.gw.detector.InterferometerList,
                 waveform_generator: bilby.gw.WaveformGenerator,
                 injection_parameters: Dict,
                 outdir: str,
                 label: str) -> bilby.result.Result:
    """
    Run Phase 1 parameter estimation with preliminary ASD.
    """
    # Set up standard priors
    priors = bilby.gw.prior.BBHPriorDict()
    priors['chirp_mass'] = Uniform(25.0, 35.0, latex_label='$\\mathcal{M}$')
    priors['mass_ratio'] = Uniform(0.5, 1.0, latex_label='$q$')
    priors['luminosity_distance'] = Uniform(200, 800, latex_label='$d_L$')
    priors['theta_jn'] = Sine(latex_label='$\\theta_{JN}$')
    priors['phase'] = Uniform(0, 2 * np.pi, latex_label='$\\phi$')
    priors['geocent_time'] = Uniform(
        injection_parameters['geocent_time'] - 0.1,
        injection_parameters['geocent_time'] + 0.1,
        latex_label='$t_c$'
    )
    
    # Fix other parameters
    for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra', 'dec', 'psi']:
        priors[key] = injection_parameters[key]
    
    # Set up likelihood
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator
    )
    
    # Sampler settings for Phase 1
    sampler_kwargs = {
        'npoints': 500,
        'walks': 25,
        'dlogz': 0.1,
        'sample': 'rwalk',
        'bound': 'multi'
    }
    
    # Run sampler
    logging.info(f"Starting Phase 1 PE: {label}")
    start_time = time.time()
    
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        outdir=outdir,
        label=label,
        injection_parameters=injection_parameters,
        save=True,
        **sampler_kwargs
    )
    
    runtime = time.time() - start_time
    logging.info(f"Phase 1 PE completed in {runtime:.2f} seconds")
    
    return result


def Create_Comparison_Plots(results: Dict, injection_parameters: Dict, outdir: str):
    """Create comprehensive comparison plots."""
    
    # Parameters to plot
    params = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'theta_jn']
    labels = {
        'chirp_mass': '$\\mathcal{M}$ $(M_\\odot)$',
        'mass_ratio': '$q$',
        'luminosity_distance': '$d_L$ (Mpc)',
        'theta_jn': '$\\theta_{JN}$ (rad)'
    }
    
    # 1. Individual parameter comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = {'phase1': 'blue', 'refined': 'green', 'baseline': 'red'}
    
    for idx, param in enumerate(params):
        ax = axes[idx]
        
        for label, result in results.items():
            if param in result.posterior:
                samples = result.posterior[param].values
                ax.hist(samples, bins=50, density=True, alpha=0.6, 
                       label=label.capitalize(), color=colors.get(label, 'gray'))
        
        # Add true value
        if param in injection_parameters:
            ax.axvline(injection_parameters[param], color='black', 
                      linestyle='--', linewidth=2, label='True')
        
        ax.set_xlabel(labels[param])
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Recovery Comparison: Phase 1 vs Refined vs Baseline', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'parameter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Comparison plots saved successfully")


# ============================================================================
# SECTION 4: MAIN REFINEMENT TEST
# ============================================================================

def Run_Complete_Refinement_Test(base_asd_files: Dict[str, str], 
                                injection_parameters: Dict = None,
                                outdir: str = None):
    """
    Run a complete refinement test including baseline comparison.
    
    Args:
        base_asd_files: Dictionary with paths to H1 and L1 ASD files
        injection_parameters: True injection parameters (if None, use defaults)
        outdir: Output directory (if None, use default path)
    """
    # Set default output directory
    if outdir is None:
        outdir = '/home/useradd/projects/bilby/MyStuff/my_outdir/phase_2/simple_refinement_test'
    
    os.makedirs(outdir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(outdir, 'test.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting complete refinement test")
    logging.info(f"Output directory: {outdir}")
    
    # Use default parameters if not provided
    if injection_parameters is None:
        injection_parameters = Create_Test_Injection_Parameters()
    
    # Initialize ASD models with just 2 stages (preliminary and final)
    asd_models = {}
    for detector, asd_file in base_asd_files.items():
        asd_models[detector] = ASD_Evolution_Model(asd_file, evolution_stages=2)
    
    # Store results
    results = {}
    
    try:
        # === PHASE 1: Initial PE with preliminary ASD ===
        logging.info("\n=== PHASE 1: Initial PE with preliminary ASD ===")
        
        # Generate preliminary ASDs (stage 0)
        preliminary_asds = {}
        for detector, model in asd_models.items():
            preliminary_asds[detector] = model.Generate_Evolved_ASD(0)
        
        # Set up interferometers
        ifos_preliminary, waveform_gen = Setup_Interferometers_With_ASD(
            preliminary_asds, injection_parameters
        )
        
        # Run Phase 1 PE
        phase1_result = Run_Phase1_PE(
            ifos_preliminary, 
            waveform_gen,
            injection_parameters,
            outdir,
            'phase1_preliminary'
        )
        results['phase1'] = phase1_result
        
        # === PHASE 2: Refined PE with final ASD ===
        logging.info("\n=== PHASE 2: Refined PE with final ASD ===")
        
        # Generate final ASDs (stage 1)
        final_asds = {}
        for detector, model in asd_models.items():
            final_asds[detector] = model.Generate_Evolved_ASD(1)
        
        # Set up interferometers with final ASD
        ifos_final, waveform_gen_final = Setup_Interferometers_With_ASD(
            final_asds, injection_parameters
        )
        
        # Create informed priors from Phase 1
        refinement_engine = PE_Refinement_Engine(outdir)
        informed_priors = refinement_engine.Create_Prior_From_Posterior(phase1_result)
        
        # Copy fixed parameters
        for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra', 'dec', 'psi']:
            informed_priors[key] = injection_parameters[key]
        
        # Run refined PE
        refined_result = refinement_engine.Run_Refined_PE(
            ifos_final,
            waveform_gen_final,
            informed_priors,
            injection_parameters,
            'phase2_refined'
        )
        results['refined'] = refined_result
        
        # === BASELINE: Traditional PE from scratch with final ASD ===
        logging.info("\n=== BASELINE: Traditional PE from scratch ===")
        
        baseline_result = Run_Phase1_PE(
            ifos_final,
            waveform_gen_final,
            injection_parameters,
            outdir,
            'baseline_from_scratch'
        )
        results['baseline'] = baseline_result
        
        # Create comparison plots
        Create_Comparison_Plots(results, injection_parameters, outdir)
        
        # Save summary
        summary = {
            'injection_parameters': injection_parameters,
            'phase1_runtime': phase1_result.sampling_time if hasattr(phase1_result, 'sampling_time') else None,
            'refined_runtime': refined_result.sampling_time if hasattr(refined_result, 'sampling_time') else None,
            'baseline_runtime': baseline_result.sampling_time if hasattr(baseline_result, 'sampling_time') else None,
        }
        
        with open(os.path.join(outdir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        logging.info("\nTest completed successfully!")
        logging.info(f"Results saved to: {outdir}")
        
        return results
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# SECTION 5: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # UPDATE THESE PATHS TO YOUR ACTUAL ASD FILES
    base_asd_files = {
        'H1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/H1_asd_win86400.pkl',
        'L1': '/home/useradd/projects/bilby/MyStuff/my_outdir/GW_Noise_H1_L1_window/L1_asd_win86400.pkl'
    }
    
    # Check if files exist
    for detector, filepath in base_asd_files.items():
        if not os.path.exists(filepath):
            print(f"WARNING: {detector} ASD file not found at: {filepath}")
            print("Please update the path to your actual ASD file")
    
    # Run complete test with baseline
    print("Starting Phase 2 Refinement Test...")
    results = Run_Complete_Refinement_Test(base_asd_files)