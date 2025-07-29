"""
Visualization suite for metabolomics priors.

This module provides comprehensive visualization capabilities for:
1. Markov field priors from metabolic reaction networks
2. Bayesian priors with uncertainty quantification
3. Chemical similarity networks
4. Prior distributions and confidence intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

# Try to import optional dependencies
try:
    import networkx as nx

    HAS_NETWORKX = True
except (ImportError, Exception):
    # Catch both ImportError and runtime errors (like the gcd import issue)
    HAS_NETWORKX = False
    nx = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


class MarkovFieldVisualizer:
    """
    Visualizer for Markov field priors derived from metabolic reaction networks.

    Handles signed metabolic networks where:
    - Positive edges connect metabolites on the same side of reactions
    - Negative edges connect substrate-product pairs
    """

    def __init__(self, signed_edges: Dict[Tuple[str, str], int]):
        """
        Initialize with signed edge dictionary.

        Args:
            signed_edges: Dictionary mapping (metabolite_i, metabolite_j) -> sign (-1 or +1)
        """
        self.signed_edges = signed_edges
        self.metabolites = list(set([m for edge in signed_edges.keys() for m in edge]))
        self.laplacian_matrix = None
        self.graph = None

        if HAS_NETWORKX:
            self._build_networkx_graph()

    def _build_networkx_graph(self):
        """Build NetworkX graph from signed edges."""
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for graph visualization")

        self.graph = nx.Graph()

        # Add nodes
        self.graph.add_nodes_from(self.metabolites)

        # Add edges with sign attributes
        for (met1, met2), sign in self.signed_edges.items():
            if met1 in self.metabolites and met2 in self.metabolites:
                self.graph.add_edge(met1, met2, sign=sign, weight=abs(sign))

    def plot_network(
        self,
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 300,
        layout: str = "spring",
        show_labels: bool = True,
        edge_width_factor: float = 1.0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the signed metabolic network.

        Args:
            figsize: Figure size tuple
            node_size: Size of metabolite nodes
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            show_labels: Whether to show metabolite labels
            edge_width_factor: Factor to scale edge widths
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for network visualization")

        fig, ax = plt.subplots(figsize=figsize)

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # Separate positive and negative edges
        positive_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d["sign"] > 0
        ]
        negative_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d["sign"] < 0
        ]

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color="lightblue",
            node_size=node_size,
            alpha=0.7,
            ax=ax,
        )

        # Draw positive edges (same-side relationships)
        if positive_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=positive_edges,
                edge_color="green",
                width=2 * edge_width_factor,
                alpha=0.6,
                style="solid",
                ax=ax,
            )

        # Draw negative edges (substrate-product relationships)
        if negative_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=negative_edges,
                edge_color="red",
                width=2 * edge_width_factor,
                alpha=0.6,
                style="dashed",
                ax=ax,
            )

        # Draw labels
        if show_labels:
            # Truncate long labels
            labels = {
                node: node[:15] + "..." if len(node) > 15 else node
                for node in self.graph.nodes()
            }
            nx.draw_networkx_labels(
                self.graph, pos, labels, font_size=8, font_weight="bold", ax=ax
            )

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color="green", lw=2, label="Same-side (+1)"),
            plt.Line2D(
                [0],
                [0],
                color="red",
                lw=2,
                linestyle="--",
                label="Substrate-Product (-1)",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        ax.set_title(
            "Signed Metabolic Network\n(Markov Field Prior Structure)",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_laplacian_heatmap(
        self,
        metabolites: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot heatmap of the graph Laplacian matrix.

        Args:
            metabolites: Subset of metabolites to include
            figsize: Figure size tuple
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        from .priors.graph_priors import build_laplacian_matrix

        if metabolites is None:
            metabolites = self.metabolites[:50]  # Limit to first 50 for readability

        # Build edges for subset
        subset_edges = [
            (m1, m2)
            for (m1, m2) in self.signed_edges.keys()
            if m1 in metabolites and m2 in metabolites
        ]

        # Build Laplacian matrix
        laplacian = build_laplacian_matrix(metabolites, subset_edges)

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Truncate metabolite names for display
        display_names = [
            name[:10] + "..." if len(name) > 10 else name for name in metabolites
        ]

        im = ax.imshow(laplacian, cmap="RdBu_r", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(range(len(metabolites)))
        ax.set_yticks(range(len(metabolites)))
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        ax.set_yticklabels(display_names)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Laplacian Value", rotation=270, labelpad=20)

        ax.set_title(
            "Graph Laplacian Matrix\n(Markov Field Prior Precision)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_degree_distribution(
        self, figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot degree distribution of the metabolic network.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for degree distribution")

        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(degree_values, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Degree Distribution")
        ax1.grid(True, alpha=0.3)

        # Log-log plot
        degree_counts = pd.Series(degree_values).value_counts().sort_index()
        ax2.loglog(degree_counts.index, degree_counts.values, "o-", alpha=0.7)
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Count")
        ax2.set_title("Degree Distribution (Log-Log)")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            "Metabolic Network Degree Analysis", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        if not HAS_NETWORKX:
            return {"error": "NetworkX required for network statistics"}

        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "positive_edges": len(
                [(u, v) for u, v, d in self.graph.edges(data=True) if d["sign"] > 0]
            ),
            "negative_edges": len(
                [(u, v) for u, v, d in self.graph.edges(data=True) if d["sign"] < 0]
            ),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "num_connected_components": nx.number_connected_components(self.graph),
        }

        if nx.is_connected(self.graph):
            stats["average_shortest_path"] = nx.average_shortest_path_length(self.graph)
            stats["diameter"] = nx.diameter(self.graph)

        return stats


class LLMPriorVisualizer:
    """
    Visualizer for Bayesian metabolite priors.

    Handles visualization of metabolite importance scores, regulation directions,
    and uncertainty quantification for Bayesian modeling.
    """

    def __init__(self, prior_scores: List[Any]):
        """
        Initialize with Bayesian metabolite prior scores.

        Args:
            prior_scores: List of metabolite score objects with Bayesian prior information
        """
        self.prior_scores = prior_scores
        self.scores_df = self._create_scores_dataframe()

    def _create_scores_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from prior scores for easier manipulation."""
        data = []
        for score in self.prior_scores:
            data.append(
                {
                    "metabolite": score.metabolite,
                    "importance_score": score.score,
                    "direction": score.direction,
                    "expected_log2fc": getattr(score, "expected_log2fc", 0.0),
                    "prior_sd": getattr(score, "prior_sd", 0.5),
                    "magnitude": getattr(score, "magnitude", "moderate"),
                    "confidence": getattr(score, "confidence", "moderate"),
                    "rationale": score.rationale,
                }
            )

        return pd.DataFrame(data)

    def plot_importance_scores(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot metabolite importance scores with direction colors.

        Args:
            top_n: Number of top metabolites to show
            figsize: Figure size tuple
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        # Get top N metabolites by importance
        top_metabolites = self.scores_df.nlargest(top_n, "importance_score")

        # Create color mapping for directions
        direction_colors = {
            "increase": "red",
            "decrease": "blue",
            "minimal": "gray",
            "unclear": "orange",
        }

        colors = [direction_colors.get(d, "gray") for d in top_metabolites["direction"]]

        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal bar plot
        bars = ax.barh(
            range(len(top_metabolites)),
            top_metabolites["importance_score"],
            color=colors,
            alpha=0.7,
        )

        # Customize plot
        ax.set_yticks(range(len(top_metabolites)))
        ax.set_yticklabels(
            [
                name[:20] + "..." if len(name) > 20 else name
                for name in top_metabolites["metabolite"]
            ]
        )
        ax.set_xlabel("Bayesian Prior Importance Score")
        ax.set_title(
            "Top Metabolites by Bayesian Prior Importance\n(Colored by Expected Direction)",
            fontsize=14,
            fontweight="bold",
        )

        # Add legend
        legend_elements = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=color, alpha=0.7, label=direction.title()
            )
            for direction, color in direction_colors.items()
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        # Add value labels on bars
        for i, (bar, score) in enumerate(
            zip(bars, top_metabolites["importance_score"])
        ):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_prior_distributions(
        self,
        metabolites: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot prior distributions for expected log2 fold changes.

        Args:
            metabolites: Specific metabolites to show (default: top 12 by importance)
            figsize: Figure size tuple
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        if metabolites is None:
            # Select top 12 by importance
            metabolites = self.scores_df.nlargest(12, "importance_score")[
                "metabolite"
            ].tolist()

        subset_df = self.scores_df[self.scores_df["metabolite"].isin(metabolites)]

        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()

        for i, (_, row) in enumerate(subset_df.iterrows()):
            if i >= 12:  # Limit to 12 plots
                break

            ax = axes[i]

            # Generate normal distribution
            x = np.linspace(
                row["expected_log2fc"] - 3 * row["prior_sd"],
                row["expected_log2fc"] + 3 * row["prior_sd"],
                100,
            )
            y = (1 / np.sqrt(2 * np.pi * row["prior_sd"] ** 2)) * np.exp(
                -0.5 * ((x - row["expected_log2fc"]) / row["prior_sd"]) ** 2
            )

            # Color based on direction
            direction_colors = {
                "increase": "red",
                "decrease": "blue",
                "minimal": "gray",
                "unclear": "orange",
            }
            color = direction_colors.get(row["direction"], "gray")

            ax.plot(x, y, color=color, linewidth=2)
            ax.fill_between(x, y, alpha=0.3, color=color)
            ax.axvline(row["expected_log2fc"], color="black", linestyle="--", alpha=0.7)
            ax.set_title(
                f"{row['metabolite'][:15]}\n{row['direction']} ({row['confidence']})",
                fontsize=10,
            )
            ax.set_xlabel("Expected Log2 FC")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, 12):
            axes[j].set_visible(False)

        plt.suptitle(
            "Prior Distributions for Expected Log2 Fold Changes",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_confidence_analysis(
        self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confidence and magnitude analysis of Bayesian priors.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Confidence distribution
        confidence_counts = self.scores_df["confidence"].value_counts()
        ax1.pie(
            confidence_counts.values,
            labels=confidence_counts.index,
            autopct="%1.1f%%",
            colors=["lightgreen", "yellow", "lightcoral"],
        )
        ax1.set_title("Confidence Distribution")

        # Magnitude vs. Importance scatter
        magnitude_order = ["minimal", "small", "moderate", "large"]
        magnitude_numeric = self.scores_df["magnitude"].map(
            {m: i for i, m in enumerate(magnitude_order)}
        )

        scatter = ax2.scatter(
            magnitude_numeric,
            self.scores_df["importance_score"],
            c=self.scores_df["prior_sd"],
            cmap="viridis",
            alpha=0.7,
        )
        ax2.set_xlabel("Magnitude")
        ax2.set_ylabel("Importance Score")
        ax2.set_title("Magnitude vs. Importance\n(Color = Prior SD)")
        ax2.set_xticks(range(len(magnitude_order)))
        ax2.set_xticklabels(magnitude_order)

        plt.colorbar(scatter, ax=ax2, label="Prior SD")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_direction_analysis(
        self, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive direction analysis.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Direction distribution
        direction_counts = self.scores_df["direction"].value_counts()
        colors = ["red", "blue", "gray", "orange"]
        ax1.pie(
            direction_counts.values,
            labels=direction_counts.index,
            autopct="%1.1f%%",
            colors=colors[: len(direction_counts)],
        )
        ax1.set_title("Direction Distribution")

        # Direction vs. Importance
        direction_order = ["decrease", "minimal", "unclear", "increase"]
        for i, direction in enumerate(direction_order):
            subset = self.scores_df[self.scores_df["direction"] == direction]
            if not subset.empty:
                ax2.scatter(
                    [i] * len(subset),
                    subset["importance_score"],
                    alpha=0.6,
                    s=50,
                    label=direction,
                )

        ax2.set_xlabel("Direction")
        ax2.set_ylabel("Importance Score")
        ax2.set_title("Importance Score by Direction")
        ax2.set_xticks(range(len(direction_order)))
        ax2.set_xticklabels(direction_order)
        ax2.legend()

        # Expected log2FC distribution
        ax3.hist(self.scores_df["expected_log2fc"], bins=20, alpha=0.7, color="skyblue")
        ax3.set_xlabel("Expected Log2 FC")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Expected Log2 FC Distribution")
        ax3.axvline(0, color="red", linestyle="--", alpha=0.7)

        # Prior SD distribution
        ax4.hist(self.scores_df["prior_sd"], bins=20, alpha=0.7, color="lightgreen")
        ax4.set_xlabel("Prior Standard Deviation")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Prior SD Distribution")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_interactive_explorer(self) -> Any:
        """
        Create interactive widget for exploring LLM priors.

        Returns:
            Interactive widget (if available)
        """
        if not HAS_WIDGETS:
            print("ipywidgets not available - install with: uv add ipywidgets")
            return None

        # Create metabolite selector
        metabolite_dropdown = widgets.Dropdown(
            options=self.scores_df["metabolite"].tolist(),
            description="Metabolite:",
            style={"description_width": "initial"},
        )

        # Create output widget
        output = widgets.Output()

        def update_plot(change):
            with output:
                output.clear_output()

                selected_metabolite = change["new"]
                row = self.scores_df[
                    self.scores_df["metabolite"] == selected_metabolite
                ].iloc[0]

                # Create detailed plot for selected metabolite
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

                # Prior distribution
                x = np.linspace(
                    row["expected_log2fc"] - 3 * row["prior_sd"],
                    row["expected_log2fc"] + 3 * row["prior_sd"],
                    100,
                )
                y = (1 / np.sqrt(2 * np.pi * row["prior_sd"] ** 2)) * np.exp(
                    -0.5 * ((x - row["expected_log2fc"]) / row["prior_sd"]) ** 2
                )

                ax1.plot(x, y, "b-", linewidth=2)
                ax1.fill_between(x, y, alpha=0.3)
                ax1.axvline(row["expected_log2fc"], color="red", linestyle="--")
                ax1.set_title(f"Prior Distribution\n{selected_metabolite}")
                ax1.set_xlabel("Expected Log2 FC")
                ax1.set_ylabel("Density")

                # Metadata display
                ax2.text(
                    0.1,
                    0.9,
                    f"Importance: {row['importance_score']:.3f}",
                    transform=ax2.transAxes,
                    fontsize=12,
                )
                ax2.text(
                    0.1,
                    0.8,
                    f"Direction: {row['direction']}",
                    transform=ax2.transAxes,
                    fontsize=12,
                )
                ax2.text(
                    0.1,
                    0.7,
                    f"Magnitude: {row['magnitude']}",
                    transform=ax2.transAxes,
                    fontsize=12,
                )
                ax2.text(
                    0.1,
                    0.6,
                    f"Confidence: {row['confidence']}",
                    transform=ax2.transAxes,
                    fontsize=12,
                )
                ax2.text(
                    0.1,
                    0.5,
                    f"Expected Log2FC: {row['expected_log2fc']:.3f}",
                    transform=ax2.transAxes,
                    fontsize=12,
                )
                ax2.text(
                    0.1,
                    0.4,
                    f"Prior SD: {row['prior_sd']:.3f}",
                    transform=ax2.transAxes,
                    fontsize=12,
                )
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.set_title("Metabolite Information")
                ax2.axis("off")

                # Rationale
                rationale_text = (
                    row["rationale"][:200] + "..."
                    if len(row["rationale"]) > 200
                    else row["rationale"]
                )
                ax3.text(
                    0.05,
                    0.95,
                    "LLM Rationale:",
                    transform=ax3.transAxes,
                    fontsize=12,
                    fontweight="bold",
                )
                ax3.text(
                    0.05,
                    0.05,
                    rationale_text,
                    transform=ax3.transAxes,
                    fontsize=10,
                    wrap=True,
                    verticalalignment="bottom",
                )
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
                ax3.axis("off")

                # Comparison with others
                comparison_df = self.scores_df[
                    self.scores_df["direction"] == row["direction"]
                ].copy()
                ax4.scatter(
                    comparison_df["expected_log2fc"],
                    comparison_df["importance_score"],
                    alpha=0.6,
                    s=50,
                    label="Same direction",
                )
                ax4.scatter(
                    [row["expected_log2fc"]],
                    [row["importance_score"]],
                    color="red",
                    s=100,
                    label="Selected",
                    zorder=5,
                )
                ax4.set_xlabel("Expected Log2 FC")
                ax4.set_ylabel("Importance Score")
                ax4.set_title(f"Comparison with {row['direction']} metabolites")
                ax4.legend()

                plt.tight_layout()
                plt.show()

        # Set up interaction
        metabolite_dropdown.observe(update_plot, names="value")

        # Initial plot
        update_plot({"new": metabolite_dropdown.value})

        return widgets.VBox([metabolite_dropdown, output])

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for LLM priors."""
        stats = {
            "num_metabolites": len(self.scores_df),
            "mean_importance": self.scores_df["importance_score"].mean(),
            "std_importance": self.scores_df["importance_score"].std(),
            "direction_distribution": self.scores_df["direction"]
            .value_counts()
            .to_dict(),
            "confidence_distribution": self.scores_df["confidence"]
            .value_counts()
            .to_dict(),
            "magnitude_distribution": self.scores_df["magnitude"]
            .value_counts()
            .to_dict(),
            "mean_expected_log2fc": self.scores_df["expected_log2fc"].mean(),
            "mean_prior_sd": self.scores_df["prior_sd"].mean(),
        }

        return stats


def create_comprehensive_report(
    markov_visualizer: MarkovFieldVisualizer,
    bayesian_visualizer: LLMPriorVisualizer,
    output_dir: str = "prior_analysis_report",
) -> None:
    """
    Create a comprehensive analysis report with all visualizations.

    Args:
        markov_visualizer: Markov field visualizer instance
        bayesian_visualizer: Bayesian prior visualizer instance
        output_dir: Directory to save the report
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("Generating comprehensive prior analysis report...")

    # Markov field visualizations
    print("1. Generating Markov field network plot...")
    markov_visualizer.plot_network(save_path=str(output_path / "markov_network.png"))

    print("2. Generating Laplacian heatmap...")
    markov_visualizer.plot_laplacian_heatmap(
        save_path=str(output_path / "laplacian_heatmap.png")
    )

    print("3. Generating degree distribution...")
    markov_visualizer.plot_degree_distribution(
        save_path=str(output_path / "degree_distribution.png")
    )

    # Bayesian prior visualizations
    print("4. Generating Bayesian importance scores...")
    bayesian_visualizer.plot_importance_scores(
        save_path=str(output_path / "bayesian_importance_scores.png")
    )

    print("5. Generating prior distributions...")
    bayesian_visualizer.plot_prior_distributions(
        save_path=str(output_path / "prior_distributions.png")
    )

    print("6. Generating confidence analysis...")
    bayesian_visualizer.plot_confidence_analysis(
        save_path=str(output_path / "confidence_analysis.png")
    )

    print("7. Generating direction analysis...")
    bayesian_visualizer.plot_direction_analysis(
        save_path=str(output_path / "direction_analysis.png")
    )

    # Generate statistics summary
    print("8. Generating statistics summary...")
    markov_stats = markov_visualizer.get_network_statistics()
    bayesian_stats = bayesian_visualizer.get_summary_statistics()

    with open(output_path / "statistics_summary.txt", "w") as f:
        f.write("APRIOMICS PRIOR ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("MARKOV FIELD NETWORK STATISTICS\n")
        f.write("-" * 30 + "\n")
        for key, value in markov_stats.items():
            f.write(f"{key}: {value}\n")

        f.write("\n\nBAYESIAN PRIOR STATISTICS\n")
        f.write("-" * 25 + "\n")
        for key, value in bayesian_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"Report generated successfully in: {output_path}")
    print("Files created:")
    for file in output_path.glob("*"):
        print(f"  - {file.name}")


# Helper functions for easy access
def plot_signed_network(
    signed_edges: Dict[Tuple[str, str], int], **kwargs
) -> plt.Figure:
    """Quick function to plot signed metabolic network."""
    visualizer = MarkovFieldVisualizer(signed_edges)
    return visualizer.plot_network(**kwargs)


def plot_bayesian_scores(prior_scores: List[Any], **kwargs) -> plt.Figure:
    """Quick function to plot Bayesian prior importance scores."""
    visualizer = LLMPriorVisualizer(prior_scores)
    return visualizer.plot_importance_scores(**kwargs)


def analyze_priors(
    signed_edges: Dict[Tuple[str, str], int],
    prior_scores: List[Any],
    output_dir: str = "prior_analysis",
) -> None:
    """
    Comprehensive analysis of both Markov field and Bayesian priors.

    Args:
        signed_edges: Dictionary of signed metabolic network edges
        prior_scores: List of Bayesian metabolite score objects
        output_dir: Output directory for analysis results
    """
    markov_viz = MarkovFieldVisualizer(signed_edges)
    bayesian_viz = LLMPriorVisualizer(prior_scores)

    create_comprehensive_report(markov_viz, bayesian_viz, output_dir)
