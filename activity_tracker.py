"""
Real-time activity tracking and visualization for bat monitoring.
Aggregates classifications into hourly bins and generates activity graphs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode for Raspberry Pi
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_LABEL_MAP = {
    "straws_fighting": ("Straws", "Fighting"),
    "rods_fighting": ("Rods", "Fighting"),
    "straws_want_food": ("Straws", "Want_Food"),
    "straws_talking": ("Straws", "Talking"),
}


def _normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_")


def _extract_species_behavior(class_name: str) -> Tuple[str, str]:
    normalized = _normalize_class_name(class_name)
    if normalized in CLASS_LABEL_MAP:
        return CLASS_LABEL_MAP[normalized]

    parts = class_name.replace("_", " ").split(maxsplit=1)
    species = parts[0] if parts else "Unknown"
    behavior = parts[1] if len(parts) > 1 else "Unknown"
    return species, behavior


class ActivityTracker:
    """
    Track bat activity in real-time with hourly aggregation.
    """
    
    def __init__(self, start_time: Optional[datetime] = None):
        """
        Initialize activity tracker.
        
        Args:
            start_time: Recording start time (default: now)
        """
        self.start_time = start_time or datetime.now()
        
        # Activity data: {hour: {class_name: count}}
        self.hourly_activity: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Raw classification list: [(timestamp, class_name, confidence)]
        self.classifications: List[Tuple[datetime, str, float]] = []
        
        # Species tracking
        self.species_counts: Dict[str, int] = defaultdict(int)
        self.behavior_counts: Dict[str, int] = defaultdict(int)
        
        # Per-species behavior tracking
        self.species_behavior_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        logger.info(f"Activity tracker initialized at {self.start_time}")
    
    def add_classification(self, class_name: str, confidence: float,
                          timestamp: Optional[datetime] = None):
        """
        Add a classification result.
        
        Args:
            class_name: Predicted class (e.g., "straws_want_food")
            confidence: Prediction confidence (0-1)
            timestamp: Classification timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store raw classification
        self.classifications.append((timestamp, class_name, confidence))
        
        # Calculate hour offset from start
        time_diff = timestamp - self.start_time
        hour = int(time_diff.total_seconds() / 3600)
        
        # Update hourly activity
        self.hourly_activity[hour][class_name] += 1
        
        # Extract species and behavior
        species = self._extract_species(class_name)
        behavior = self._extract_behavior(class_name)
        
        # Update counts
        self.species_counts[species] += 1
        self.behavior_counts[behavior] += 1
        self.species_behavior_counts[species][behavior] += 1
    
    def _extract_species(self, class_name: str) -> str:
        """Extract species from class name."""
        species, _ = _extract_species_behavior(class_name)
        return species
    
    def _extract_behavior(self, class_name: str) -> str:
        """Extract behavior from class name."""
        _, behavior = _extract_species_behavior(class_name)
        return behavior
    
    def get_hourly_activity(self) -> Dict[int, Dict[str, int]]:
        """Get hourly activity data."""
        return dict(self.hourly_activity)
    
    def get_species_counts(self) -> Dict[str, int]:
        """Get total counts per species."""
        return dict(self.species_counts)
    
    def get_behavior_counts(self) -> Dict[str, int]:
        """Get total counts per behavior."""
        return dict(self.behavior_counts)
    
    def get_species_behavior_counts(self) -> Dict[str, Dict[str, int]]:
        """Get behavior counts per species."""
        return {species: dict(behaviors) for species, behaviors in self.species_behavior_counts.items()}
    
    def get_total_vocalizations(self) -> int:
        """Get total number of vocalizations detected."""
        return len(self.classifications)
    
    def get_want_food_rate(self, species: str = "Straws") -> float:
        """
        Get Want_Food calls per hour for food consumption prediction.
        
        Args:
            species: Species name (default: "Straws")
            
        Returns:
            Average Want_Food calls per hour
        """
        # Get total Want_Food calls for this species
        want_food_count = self.species_behavior_counts.get(species, {}).get("Want_Food", 0)
        
        # Calculate recording duration in hours
        if self.classifications:
            last_time = self.classifications[-1][0]
            duration_hours = (last_time - self.start_time).total_seconds() / 3600
            duration_hours = max(duration_hours, 0.01)  # Avoid division by zero
        else:
            duration_hours = 1.0
        
        return want_food_count / duration_hours
    
    def get_recording_duration(self) -> timedelta:
        """Get total recording duration."""
        if self.classifications:
            return self.classifications[-1][0] - self.start_time
        return timedelta(0)
    
    def generate_timeline_graph(self, output_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6),
                               dpi: int = 100) -> Figure:
        """
        Generate hourly activity timeline graph.
        
        Args:
            output_path: Path to save PNG (None = don't save)
            figsize: Figure size in inches
            dpi: DPI for saved image
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Get unique species
        species_list = sorted(self.species_counts.keys())
        
        if not species_list:
            ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # Prepare data for stacked bar chart
            hours = sorted(self.hourly_activity.keys())
            if not hours:
                hours = [0]
            
            # Create data matrix: species x hours
            data_matrix = {species: [] for species in species_list}
            
            for hour in hours:
                hour_data = self.hourly_activity[hour]
                for species in species_list:
                    # Sum all behaviors for this species in this hour
                    species_count = sum(
                        count for class_name, count in hour_data.items()
                        if self._extract_species(class_name) == species
                    )
                    data_matrix[species].append(species_count)
            
            # Plot stacked bars
            bottom = np.zeros(len(hours))
            colormap = plt.cm.get_cmap('Set3')
            colors = colormap(np.linspace(0, 1, len(species_list)))
            
            for idx, species in enumerate(species_list):
                counts = data_matrix[species]
                ax.bar(hours, counts, bottom=bottom, label=species,
                      color=colors[idx], alpha=0.8)
                bottom += counts
            
            # Format plot
            ax.set_xlabel('Hour', fontsize=12)
            ax.set_ylabel('Number of Vocalizations', fontsize=12)
            ax.set_title(f'Bat Activity Timeline - {self.start_time.strftime("%Y-%m-%d")}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            
            # Set x-axis
            if len(hours) > 1:
                ax.set_xticks(hours)
            ax.set_xlabel('Hours from Start', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Timeline graph saved to {output_path}")
        
        return fig
    
    def generate_breakdown_graph(self, output_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6),
                                dpi: int = 100) -> Figure:
        """
        Generate behavior breakdown graph per species.
        
        Args:
            output_path: Path to save PNG
            figsize: Figure size
            dpi: Image DPI
            
        Returns:
            Matplotlib Figure
        """
        species_list = sorted(self.species_counts.keys())
        
        if not species_list:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            return fig
        
        # Create subplots for each species
        n_species = len(species_list)
        fig, axes = plt.subplots(1, n_species, figsize=figsize, dpi=dpi)
        
        if n_species == 1:
            axes = [axes]
        
        for idx, species in enumerate(species_list):
            ax = axes[idx]
            behaviors = self.species_behavior_counts[species]
            
            if behaviors:
                behavior_names = list(behaviors.keys())
                counts = list(behaviors.values())
                
                # Create pie chart
                colormap = plt.cm.get_cmap('Pastel1')
                colors = colormap(np.linspace(0, 1, len(behavior_names)))
                wedges, texts, autotexts = ax.pie(
                    counts,
                    labels=behavior_names,
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90
                )
                
                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontsize(9)
                    autotext.set_weight('bold')
                
                ax.set_title(f'{species}\n({sum(counts)} vocalizations)',
                           fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        
        plt.suptitle(f'Behavior Breakdown - {self.start_time.strftime("%Y-%m-%d")}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Breakdown graph saved to {output_path}")
        
        return fig
    
    def generate_summary_table(self) -> str:
        """
        Generate text summary table.
        
        Returns:
            Formatted string table
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BAT ACTIVITY SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Recording Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration: {self.get_recording_duration()}")
        lines.append(f"Total Vocalizations: {self.get_total_vocalizations()}")
        lines.append("")
        
        # Species breakdown
        lines.append("Species Counts:")
        lines.append("-" * 40)
        for species, count in sorted(self.species_counts.items()):
            percentage = (count / max(1, self.get_total_vocalizations())) * 100
            lines.append(f"  {species:20s}: {count:5d} ({percentage:5.1f}%)")
        lines.append("")
        
        # Behavior breakdown per species
        for species in sorted(self.species_behavior_counts.keys()):
            lines.append(f"{species} Behaviors:")
            lines.append("-" * 40)
            behaviors = self.species_behavior_counts[species]
            species_total = sum(behaviors.values())
            for behavior, count in sorted(behaviors.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / max(1, species_total)) * 100
                lines.append(f"  {behavior:20s}: {count:5d} ({percentage:5.1f}%)")
            
            # Want_Food rate
            if "Want_Food" in behaviors:
                rate = self.get_want_food_rate(species)
                lines.append(f"  Want_Food calls/hour: {rate:.2f}")
            
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def export_to_dict(self) -> Dict:
        """Export all data as dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'duration_seconds': self.get_recording_duration().total_seconds(),
            'total_vocalizations': self.get_total_vocalizations(),
            'species_counts': self.get_species_counts(),
            'behavior_counts': self.get_behavior_counts(),
            'species_behavior_counts': self.get_species_behavior_counts(),
            'hourly_activity': self.get_hourly_activity(),
            'classifications': [
                {
                    'timestamp': ts.isoformat(),
                    'class': cls,
                    'confidence': conf
                }
                for ts, cls, conf in self.classifications
            ]
        }


def test_activity_tracker():
    """Test activity tracking and visualization."""
    print("Testing Activity Tracker...")
    
    # Create tracker
    tracker = ActivityTracker()
    
    # Simulate classifications over several hours
    import random
    
    species_behaviors = [
        ("Rods", "Fighting"),
        ("Straws", "Fighting"),
        ("Straws", "Want_Food"),
        ("Straws", "Talking"),
    ]
    
    start = datetime.now()
    
    # Generate random data over 6 hours
    for hour in range(6):
        num_events = random.randint(10, 30)
        for _ in range(num_events):
            species, behavior = random.choice(species_behaviors)
            class_name = f"{species.lower()}_{behavior.lower()}"
            confidence = random.uniform(0.6, 0.99)
            timestamp = start + timedelta(hours=hour, minutes=random.randint(0, 59))
            
            tracker.add_classification(class_name, confidence, timestamp)
    
    # Print summary
    print(tracker.generate_summary_table())
    
    # Generate graphs
    print("\nGenerating graphs...")
    tracker.generate_timeline_graph("test_timeline.png")
    tracker.generate_breakdown_graph("test_breakdown.png")
    
    print("\n✓ Activity tracker test completed!")
    print("  Generated: test_timeline.png, test_breakdown.png")
    
    # Get Want_Food rate
    straws_rate = tracker.get_want_food_rate("Straws")
    print(f"\nStraws Want_Food rate: {straws_rate:.2f} calls/hour")


if __name__ == "__main__":
    test_activity_tracker()
