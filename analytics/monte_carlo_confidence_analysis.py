import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import logging

# Configure enterprise-grade logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [MonteCarloEdgeSim] %(message)s')
logger = logging.getLogger(__name__)

class YOLOEdgeMonteCarloSimulator:
    """
    A Monte Carlo simulation engine to evaluate the statistical reliability of 
    downstream robotic sorting logic given real-world variance in YOLOv8
    bounding box confidence scores at the edge.
    
    Substantiates: JHU EDU_JHU_05 (Applied Analytics for MBSE)
    """
    def __init__(self, n_simulations: int = 100000):
        self.n = n_simulations
        self.target_class = "Dented"
        
        # Define Ground Truth Simulation Parameters
        # In this scenario, a genuinely Dented box is on the conveyor belt.
        # We model the AI's confidence score output as a probability distribution
        # subject to environmental noise (camera blur, lighting shifts).
        
        # Mean confidence the AI outputs for this specific damaged box (from empiric validation)
        self.mu_dented = 0.82 
        # Standard deviation (variance) of the AI's confidence over time/frames
        self.sigma_dented = 0.12
        
        # If the model gets confused, it might predict "Pristine"
        self.mu_pristine = 0.15
        self.sigma_pristine = 0.08
        
        # The threshold at which the PLC/Sorting Arm actually fires the pneumatic rejector
        self.plc_reject_threshold = 0.65
        
        logger.info(f"Initialized Simulator with {self.n:,} iterations.")
        logger.info(f"Ground Truth: {self.target_class} Box")
        logger.info(f"PLC Reject Activation Threshold: >{self.plc_reject_threshold*100}% Confidence")

    def run_simulation(self):
        """Executes the stochastic Monte Carlo analysis"""
        logger.info("Generating stochastic tensor matrices...")
        
        # 1. Simulate 100k random variations of the AI's confidence output for the Dented class
        # Clip values between 0.0 and 1.0 since confidence cannot exceed 100%
        simulated_dented_conf = np.random.normal(self.mu_dented, self.sigma_dented, self.n)
        simulated_dented_conf = np.clip(simulated_dented_conf, 0.0, 1.0)
        
        # 2. Simulate the AI's confusion (predicting Pristine incorrectly)
        simulated_pristine_conf = np.random.normal(self.mu_pristine, self.sigma_pristine, self.n)
        simulated_pristine_conf = np.clip(simulated_pristine_conf, 0.0, 1.0)
        
        # 3. Decision Logic: Did the AI's output trigger the robotic arm?
        # A successful reject requires:
        # A) The Dented confidence is strictly higher than Pristine confidence
        # B) The Dented confidence exceeds the hardcoded PLC safety threshold
        
        successful_rejects = (simulated_dented_conf > simulated_pristine_conf) & (simulated_dented_conf >= self.plc_reject_threshold)
        
        # 4. Calculate final system reliability
        total_successes = np.sum(successful_rejects)
        reliability_pct = (total_successes / self.n) * 100
        
        logger.info(f"Simulation Complete. Total Successful Rejects: {total_successes:,} / {self.n:,}")
        logger.info(f"Overall System Reliability: {reliability_pct:.2f}% under real-world noise.")
        
        self._generate_plots(simulated_dented_conf, reliability_pct)

    def _generate_plots(self, dented_conf: np.ndarray, reliability: float):
        """Generates the Probability Density Function (PDF) visualization"""
        logger.info("Rendering Probability Density Function plotting...")
        
        plt.figure(figsize=(10, 6))
        
        # Plot the Histogram of the simulation results
        plt.hist(dented_conf, bins=100, density=True, alpha=0.6, color='coral', label='Simulated YOLOv8 Confidences')
        
        # Fit and plot a clean Gaussian curve over the noisy data
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, np.mean(dented_conf), np.std(dented_conf))
        plt.plot(x, p, 'k', linewidth=2, label='Fitted PDF')
        
        # Draw the physical PLC threshold boundary line
        plt.axvline(x=self.plc_reject_threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'PLC Reject Threshold (T={self.plc_reject_threshold})')
        
        # Fill the "Failure" zone (Type II Error - False Negative)
        x_fail = np.linspace(xmin, self.plc_reject_threshold, 50)
        y_fail = norm.pdf(x_fail, np.mean(dented_conf), np.std(dented_conf))
        plt.fill_between(x_fail, 0, y_fail, color='red', alpha=0.2, label='System Failure Margin')
        
        plt.title('Monte Carlo Simulation: Edge Node Inference Reliability', fontsize=14, fontweight='bold')
        plt.xlabel('AI Object Detection Confidence Score (Class: Dented)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        
        # Subtitle metrics
        metrics_text = f"Simulations: {self.n:,}\nTarget Box: Dented\nSystem Reliability: {reliability:.2f}%"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        out_path = '/home/caaren/dev/personal/package_integrity_classification_via_sim-to-real/analytics/monte_carlo_distribution.png'
        plt.savefig(out_path, dpi=300)
        logger.info(f"Plot explicitly saved to: {out_path}")
        plt.close()

if __name__ == "__main__":
    engine = YOLOEdgeMonteCarloSimulator(n_simulations=500000)
    engine.run_simulation()
    
