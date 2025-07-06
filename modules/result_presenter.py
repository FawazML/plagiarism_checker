# result_presenter.py
from typing import Set, Tuple, List, Dict, Any, Optional, Union, Callable
import json
import csv
import os
import datetime
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from colorama import init, Fore, Style
import seaborn as sns

# Initialize colorama for colored terminal output
init(autoreset=True)

class OutputFormat(Enum):
    """Supported output formats for similarity results."""
    CONSOLE = "console"
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"
    LATEX = "latex"

class ColorTheme(Enum):
    """Color themes for console output."""
    NONE = "none"  # No colors
    BASIC = "basic"  # Simple color scheme
    HEATMAP = "heatmap"  # Colors based on similarity values
    TRAFFIC_LIGHT = "traffic_light"  # Red, yellow, green based on thresholds

class ResultFormatter:
    """
    Class for formatting similarity results into various structured formats.
    """
    
    @staticmethod
    def to_dict_list(results: Set[Tuple[str, str, float]], 
                    precision: int = 4) -> List[Dict[str, Any]]:
        """
        Convert similarity results to a list of dictionaries.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            
        Returns:
            List of dictionaries with document1, document2, and similarity
        """
        formatted_results = []
        
        for doc1, doc2, similarity in results:
            formatted_results.append({
                "document1": doc1,
                "document2": doc2,
                "similarity": round(similarity, precision),
                "similarity_percentage": round(similarity * 100, precision - 2)
            })
            
        return formatted_results
    
    @staticmethod
    def to_json(results: Set[Tuple[str, str, float]], 
               precision: int = 4, 
               pretty: bool = True) -> str:
        """
        Convert similarity results to a JSON string.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            pretty: Whether to format the JSON with indentation
            
        Returns:
            JSON string representation of results
        """
        formatted_results = ResultFormatter.to_dict_list(results, precision)
        indent = 2 if pretty else None
        return json.dumps(formatted_results, indent=indent)
    
    @staticmethod
    def to_csv_str(results: Set[Tuple[str, str, float]], 
                  precision: int = 4) -> str:
        """
        Convert similarity results to a CSV string.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            
        Returns:
            CSV string representation of results
        """
        formatted_results = ResultFormatter.to_dict_list(results, precision)
        if not formatted_results:
            return "document1,document2,similarity,similarity_percentage"
        
        # Create CSV string
        output = []
        output.append("document1,document2,similarity,similarity_percentage")
        
        for item in formatted_results:
            row = f"{item['document1']},{item['document2']},{item['similarity']},{item['similarity_percentage']}"
            output.append(row)
            
        return "\n".join(output)
    
    @staticmethod
    def to_markdown_table(results: Set[Tuple[str, str, float]], 
                         precision: int = 4) -> str:
        """
        Convert similarity results to a Markdown table.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            
        Returns:
            Markdown table representation of results
        """
        formatted_results = ResultFormatter.to_dict_list(results, precision)
        if not formatted_results:
            return "No results to display."
        
        # Create markdown table
        output = []
        output.append("| Document 1 | Document 2 | Similarity | Percentage |")
        output.append("|------------|------------|------------|------------|")
        
        for item in formatted_results:
            row = f"| {item['document1']} | {item['document2']} | {item['similarity']} | {item['similarity_percentage']}% |"
            output.append(row)
            
        return "\n".join(output)
    
    @staticmethod
    def to_html_table(results: Set[Tuple[str, str, float]], 
                     precision: int = 4, 
                     with_style: bool = True) -> str:
        """
        Convert similarity results to an HTML table.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            with_style: Whether to include CSS styling
            
        Returns:
            HTML table representation of results
        """
        formatted_results = ResultFormatter.to_dict_list(results, precision)
        if not formatted_results:
            return "<p>No results to display.</p>"
        
        # Create HTML with optional styling
        css_style = """
        <style>
            .similarity-table {
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
            }
            .similarity-table th, .similarity-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .similarity-table tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .similarity-table tr:hover {
                background-color: #ddd;
            }
            .similarity-table th {
                padding-top: 12px;
                padding-bottom: 12px;
                background-color: #4CAF50;
                color: white;
            }
            .high-similarity {
                background-color: #ffcccc !important;
            }
        </style>
        """ if with_style else ""
        
        html = []
        html.append(css_style)
        html.append('<table class="similarity-table">')
        html.append('<tr><th>Document 1</th><th>Document 2</th><th>Similarity</th><th>Percentage</th></tr>')
        
        for item in sorted(formatted_results, key=lambda x: x['similarity'], reverse=True):
            # Add class for high similarity if over 0.7
            row_class = ' class="high-similarity"' if item['similarity'] > 0.7 else ''
            row = f"<tr{row_class}><td>{item['document1']}</td><td>{item['document2']}</td><td>{item['similarity']}</td><td>{item['similarity_percentage']}%</td></tr>"
            html.append(row)
            
        html.append('</table>')
        return "\n".join(html)
    
    @staticmethod
    def to_latex_table(results: Set[Tuple[str, str, float]], 
                      precision: int = 4) -> str:
        """
        Convert similarity results to a LaTeX table.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            
        Returns:
            LaTeX table representation of results
        """
        formatted_results = ResultFormatter.to_dict_list(results, precision)
        if not formatted_results:
            return "% No results to display."
        
        # Create LaTeX table
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\begin{tabular}{|l|l|r|r|}")
        latex.append("\\hline")
        latex.append("Document 1 & Document 2 & Similarity & Percentage \\\\ \\hline")
        
        for item in sorted(formatted_results, key=lambda x: x['similarity'], reverse=True):
            row = f"{item['document1']} & {item['document2']} & {item['similarity']} & {item['similarity_percentage']}\\% \\\\ \\hline"
            latex.append(row)
            
        latex.append("\\end{tabular}")
        latex.append("\\caption{Document Similarity Results}")
        latex.append("\\end{table}")
        return "\n".join(latex)
    
    @staticmethod
    def to_matrix_format(results: Set[Tuple[str, str, float]], 
                        precision: int = 4) -> Tuple[np.ndarray, List[str]]:
        """
        Convert similarity results to a similarity matrix.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            
        Returns:
            Tuple of similarity matrix and document names
        """
        # Get all unique document names
        doc_names = set()
        for doc1, doc2, _ in results:
            doc_names.add(doc1)
            doc_names.add(doc2)
        
        # Sort document names for consistent ordering
        doc_names = sorted(list(doc_names))
        n_docs = len(doc_names)
        
        # Create a mapping from document name to index
        doc_to_idx = {name: i for i, name in enumerate(doc_names)}
        
        # Initialize similarity matrix with diagonal = 1
        sim_matrix = np.eye(n_docs, dtype=float)
        
        # Fill in similarity values
        for doc1, doc2, similarity in results:
            i, j = doc_to_idx[doc1], doc_to_idx[doc2]
            sim_matrix[i, j] = round(similarity, precision)
            sim_matrix[j, i] = round(similarity, precision)  # Mirror for symmetry
        
        return sim_matrix, doc_names
    
    @staticmethod
    def to_pandas_dataframe(results: Set[Tuple[str, str, float]], 
                          precision: int = 4) -> pd.DataFrame:
        """
        Convert similarity results to a pandas DataFrame.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            
        Returns:
            pandas DataFrame with results
        """
        return pd.DataFrame(ResultFormatter.to_dict_list(results, precision))
    
    @staticmethod
    def to_similarity_matrix_dataframe(results: Set[Tuple[str, str, float]], 
                                     precision: int = 4) -> pd.DataFrame:
        """
        Convert similarity results to a pandas DataFrame in matrix format.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            
        Returns:
            pandas DataFrame with similarity matrix
        """
        sim_matrix, doc_names = ResultFormatter.to_matrix_format(results, precision)
        return pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
        
    @staticmethod
    def to_tabulate(results: Set[Tuple[str, str, float]], 
                   precision: int = 4, 
                   table_format: str = 'simple') -> str:
        """
        Format results using tabulate for pretty console tables.
        
        Args:
            results: Set of tuples with document pairs and similarities
            precision: Number of decimal places for similarity values
            table_format: Table format for tabulate (grid, simple, fancy_grid, etc)
            
        Returns:
            Formatted table string
        """
        formatted_results = ResultFormatter.to_dict_list(results, precision)
        
        # Prepare data for tabulate
        headers = ["Document 1", "Document 2", "Similarity", "Percentage"]
        data = []
        
        for item in sorted(formatted_results, key=lambda x: x['similarity'], reverse=True):
            data.append([
                item['document1'], 
                item['document2'], 
                item['similarity'], 
                f"{item['similarity_percentage']}%"
            ])
        
        return tabulate(data, headers=headers, tablefmt=table_format)


class ConsoleFormatter:
    """
    Class for formatting and colorizing console output.
    """
    
    @staticmethod
    def color_by_similarity(similarity: float) -> str:
        """
        Return color code based on similarity value.
        
        Args:
            similarity: Similarity value between 0 and 1
            
        Returns:
            ANSI color code
        """
        if similarity >= 0.8:
            return Fore.RED  # High similarity (potential plagiarism)
        elif similarity >= 0.5:
            return Fore.YELLOW  # Medium similarity
        elif similarity >= 0.3:
            return Fore.GREEN  # Low similarity
        else:
            return Fore.BLUE  # Very low similarity
    
    @staticmethod
    def colorize_result(doc1: str, doc2: str, similarity: float, 
                      precision: int = 4, 
                      color_theme: ColorTheme = ColorTheme.HEATMAP) -> str:
        """
        Format a single result with color based on the theme.
        
        Args:
            doc1: First document name
            doc2: Second document name
            similarity: Similarity score
            precision: Decimal precision
            color_theme: Color theme to use
            
        Returns:
            Colored string for console output
        """
        sim_formatted = f"{similarity:.{precision}f}"
        percentage = f"{similarity * 100:.{precision-2}f}%"
        
        if color_theme == ColorTheme.NONE:
            return f"{doc1} and {doc2}: {sim_formatted} similarity ({percentage})"
            
        elif color_theme == ColorTheme.BASIC:
            return f"{Fore.CYAN}{doc1}{Style.RESET_ALL} and {Fore.CYAN}{doc2}{Style.RESET_ALL}: " \
                   f"{Fore.YELLOW}{sim_formatted}{Style.RESET_ALL} similarity ({Fore.YELLOW}{percentage}{Style.RESET_ALL})"
                   
        elif color_theme == ColorTheme.HEATMAP:
            color = ConsoleFormatter.color_by_similarity(similarity)
            return f"{Fore.CYAN}{doc1}{Style.RESET_ALL} and {Fore.CYAN}{doc2}{Style.RESET_ALL}: " \
                   f"{color}{sim_formatted}{Style.RESET_ALL} similarity ({color}{percentage}{Style.RESET_ALL})"
                   
        elif color_theme == ColorTheme.TRAFFIC_LIGHT:
            if similarity >= 0.7:
                color = Fore.RED
                level = "HIGH"
            elif similarity >= 0.4:
                color = Fore.YELLOW
                level = "MEDIUM"
            else:
                color = Fore.GREEN
                level = "LOW"
                
            return f"{Fore.CYAN}{doc1}{Style.RESET_ALL} and {Fore.CYAN}{doc2}{Style.RESET_ALL}: " \
                   f"{color}{sim_formatted}{Style.RESET_ALL} similarity ({color}{percentage}{Style.RESET_ALL}) - " \
                   f"{color}{level}{Style.RESET_ALL} similarity level"
        
        return f"{doc1} and {doc2}: {sim_formatted} similarity ({percentage})"


class ResultVisualizer:
    """
    Class for visualizing similarity results.
    """
    
    @staticmethod
    def plot_similarity_heatmap(results: Set[Tuple[str, str, float]], 
                              output_file: Optional[str] = None,
                              title: str = "Document Similarity Heatmap",
                              cmap: str = "viridis",
                              figsize: Tuple[int, int] = (12, 10),
                              show: bool = True) -> None:
        """
        Create a heatmap visualization of the similarity matrix using seaborn.
        
        Args:
            results: Set of tuples with document pairs and similarities
            output_file: Optional path to save the plot
            title: Plot title
            cmap: Colormap to use
            figsize: Figure size (width, height)
            show: Whether to display the plot (use False for headless environments)
        """
        # Convert results to matrix format and then to DataFrame for seaborn
        sim_matrix, doc_names = ResultFormatter.to_matrix_format(results)
        df_matrix = pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)
        
        # Set seaborn style and context for better aesthetics
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.4)
        
        # Set up the figure with adjusted size for better readability
        plt.figure(figsize=figsize)
        
        # Create a mask for the diagonal if you want to hide self-similarity (always 1.0)
        mask = np.zeros_like(sim_matrix, dtype=bool)
        np.fill_diagonal(mask, True)  # True values will be masked
        
        # Custom colormap - use a predefined one or create a custom gradient
        if cmap == "custom_heat":
            cmap = sns.color_palette("YlOrRd", as_cmap=True)
        elif cmap == "custom_blue_red":
            # Create a custom diverging colormap from blue to red
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Create the heatmap using seaborn with enhanced aesthetics
        ax = sns.heatmap(
            df_matrix,
            annot=True,  # Show the values within cells
            fmt=".2f",   # Format for annotations (2 decimal places)
            cmap=cmap,
            mask=mask,   # Apply the mask to hide diagonal values
            linewidths=0.7,
            cbar_kws={
                "label": "Similarity Score", 
                "shrink": 0.8,
                "aspect": 20,
                "pad": 0.12
            },
            square=True,  # Ensure cells are square
            annot_kws={"size": 12, "weight": "bold"},
            vmin=0,  # Set minimum value for consistent color scaling
            vmax=1   # Set maximum value for consistent color scaling
        )
        
        # Enhance the title and axis labels
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Documents", fontsize=14, fontweight="bold", labelpad=15)
        plt.ylabel("Documents", fontsize=14, fontweight="bold", labelpad=15)
        
        # Rotate x-axis labels for better readability and adjust font properties
        plt.xticks(rotation=45, ha="right", fontsize=12, fontweight="bold")
        plt.yticks(fontsize=12, fontweight="bold")
        
        # Add a custom border to the heatmap
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        
        # Add custom annotations for high similarity pairs (>=0.7)
        threshold_colors = {
            0.9: "#8B0000",  # Dark red for very high similarity (≥0.9)
            0.7: "#FF4500",  # Red/Orange for high similarity (≥0.7)
        }
        
        for i in range(len(doc_names)):
            for j in range(len(doc_names)):
                if i != j:  # Skip diagonal
                    similarity = sim_matrix[i, j]
                    # Add colored borders based on threshold level
                    if similarity >= 0.9:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                   edgecolor=threshold_colors[0.9], 
                                                   lw=3, clip_on=False))
                    elif similarity >= 0.7:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                   edgecolor=threshold_colors[0.7], 
                                                   lw=2, clip_on=False))
        
        # Add a legend for the threshold indicators
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor=threshold_colors[0.9], 
                  label='Very High Similarity (≥0.9)', linewidth=3),
            Patch(facecolor='none', edgecolor=threshold_colors[0.7], 
                  label='High Similarity (≥0.7)', linewidth=2)
        ]
        ax.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), frameon=True, ncol=2, 
                  fontsize=11, title="Similarity Thresholds", title_fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for legend space
        
        # Save if output file is provided
        if output_file:
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_similarity_bar_chart(results: Set[Tuple[str, str, float]], 
                                output_file: Optional[str] = None,
                                title: str = "Document Similarity Scores",
                                figsize: Tuple[int, int] = (14, 8),
                                threshold: float = 0.0,
                                show: bool = True) -> None:
        """
        Create a bar chart of similarity scores with enhanced aesthetics.
        
        Args:
            results: Set of tuples with document pairs and similarities
            output_file: Optional path to save the plot
            title: Plot title
            figsize: Figure size (width, height)
            threshold: Only show pairs with similarity above this threshold
            show: Whether to display the plot (use False for headless environments)
        """
        # Filter results by threshold
        filtered_results = [(doc1, doc2, sim) for doc1, doc2, sim in results if sim >= threshold]
        
        # Sort by similarity descending
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        
        # Extract data for plotting
        pairs = [f"{doc1}\\n{doc2}" for doc1, doc2, _ in filtered_results]
        scores = [sim for _, _, sim in filtered_results]
        
        # Create a pandas DataFrame for seaborn
        df = pd.DataFrame({
            "Document Pairs": pairs,
            "Similarity Score": scores
        })
        
        # Define color palette based on similarity threshold
        def get_color(sim):
            if sim >= 0.9: return "#8B0000"      # Very high similarity
            elif sim >= 0.7: return "#FF4500"    # High similarity
            elif sim >= 0.5: return "#FF8C00"    # Moderate similarity
            elif sim >= 0.3: return "#1E90FF"    # Low similarity
            else: return "#4682B4"               # Very low similarity
        
        colors = [get_color(sim) for sim in scores]
        
        # Set seaborn style and context for better aesthetics
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.4)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Use seaborn barplot with improved aesthetics
        ax = sns.barplot(
            x="Document Pairs",
            y="Similarity Score",
            data=df,
            palette=colors,
            edgecolor="black",
            linewidth=1.5
        )
        
        # Add value labels on top of bars
        for i, score in enumerate(scores):
            ax.text(
                i,
                score + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
            )
        
        # Customize the plot
        plt.title(title, fontsize=18, fontweight="bold", pad=20)
        plt.xlabel("Document Pairs", fontsize=15, fontweight="bold", labelpad=15)
        plt.ylabel("Similarity Score", fontsize=15, fontweight="bold", labelpad=15)
        plt.ylim(0, 1.15)  # Set y-axis limit to allow space for labels
        
        # Add grid lines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add threshold lines with labels
        plt.axhline(y=0.9, color="#8B0000", linestyle="--", alpha=0.7, linewidth=2, label="Very High (≥0.9)")
        plt.axhline(y=0.7, color="#FF4500", linestyle="--", alpha=0.7, linewidth=2, label="High (≥0.7)")
        plt.axhline(y=0.5, color="#FF8C00", linestyle="--", alpha=0.5, linewidth=2, label="Moderate (≥0.5)")
        
        # Create a styled legend
        legend = plt.legend(
            loc="upper right", 
            title="Similarity Thresholds",
            frameon=True, 
            fancybox=True, 
            shadow=True, 
            fontsize=12,
            title_fontsize=14
        )
        
        # Style the axis ticks
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add a special annotation for the highest similarity pair
        if scores:
            max_idx = scores.index(max(scores))
            ax.get_children()[max_idx].set_edgecolor('black')
            ax.get_children()[max_idx].set_linewidth(2.5)
            
            # Add a text box highlighting the most similar pair
            highest_pair = pairs[max_idx].replace('\\n', ' & ')
            plt.annotate(
                f"Most similar: {highest_pair} ({scores[max_idx]:.2f})",
                xy=(max_idx, scores[max_idx]),
                xytext=(max_idx, scores[max_idx] + 0.15),
                fontsize=13,
                fontweight="bold",
                ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='black'),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black')
            )
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save if output file is provided
        if output_file:
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()

class ResultPresenter:
    """
    Main class for presenting similarity results in various formats.
    """
    
    def __init__(self,
                console_color_theme: ColorTheme = ColorTheme.HEATMAP,
                precision: int = 4,
                default_output_format: OutputFormat = OutputFormat.CONSOLE,
                default_table_format: str = "grid"):
        """
        Initialize ResultPresenter with default settings.
        
        Args:
            console_color_theme: Color theme for console output
            precision: Decimal precision for similarity values
            default_output_format: Default output format
            default_table_format: Default table format for tabulate
        """
        self.console_color_theme = console_color_theme
        self.precision = precision
        self.default_output_format = default_output_format
        self.default_table_format = default_table_format
        self.formatter = ResultFormatter()
        self.console_formatter = ConsoleFormatter()
        self.visualizer = ResultVisualizer()
        
    def print_results(self, results: Set[Tuple[str, str, float]], 
                     sort_by_similarity: bool = True,
                     reverse: bool = True) -> None:
        """
        Print similarity results to the console with color formatting.
        
        Args:
            results: Set of tuples with document pairs and similarities
            sort_by_similarity: Whether to sort results by similarity score
            reverse: Whether to sort in descending order
        """
        # Convert to list and sort if requested
        result_list = list(results)
        if sort_by_similarity:
            result_list.sort(key=lambda x: x[2], reverse=reverse)
            
        # Print each result
        for doc1, doc2, similarity in result_list:
            colored_line = ConsoleFormatter.colorize_result(
                doc1, doc2, similarity, 
                precision=self.precision, 
                color_theme=self.console_color_theme
            )
            print(colored_line)
    
    def print_summary(self, results: Set[Tuple[str, str, float]]) -> None:
        """
        Print a summary of similarity results.
        
        Args:
            results: Set of tuples with document pairs and similarities
        """
        if not results:
            print(f"{Fore.YELLOW}No similarity results available.{Style.RESET_ALL}")
            return
            
        # Extract all documents and similarities
        all_docs = set()
        similarities = []
        
        for doc1, doc2, sim in results:
            all_docs.add(doc1)
            all_docs.add(doc2)
            similarities.append(sim)
        
        # Calculate statistics
        num_docs = len(all_docs)
        num_comparisons = len(results)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        min_similarity = min(similarities) if similarities else 0
        
        # Get the pair with maximum similarity
        max_pair = None
        for doc1, doc2, sim in results:
            if sim == max_similarity:
                max_pair = (doc1, doc2)
                break
        
        # Print summary with colors
        print(f"\n{Fore.CYAN}===== Similarity Analysis Summary ====={Style.RESET_ALL}")
        print(f"Number of documents: {Fore.GREEN}{num_docs}{Style.RESET_ALL}")
        print(f"Number of comparisons: {Fore.GREEN}{num_comparisons}{Style.RESET_ALL}")
        print(f"Average similarity: {Fore.YELLOW}{avg_similarity:.4f}{Style.RESET_ALL}")
        print(f"Maximum similarity: {ConsoleFormatter.color_by_similarity(max_similarity)}{max_similarity:.4f}{Style.RESET_ALL}")
        if max_pair:
            print(f"Most similar pair: {Fore.CYAN}{max_pair[0]}{Style.RESET_ALL} and {Fore.CYAN}{max_pair[1]}{Style.RESET_ALL}")
        print(f"Minimum similarity: {ConsoleFormatter.color_by_similarity(min_similarity)}{min_similarity:.4f}{Style.RESET_ALL}")
        
        # Add high similarity warning
        high_sim_pairs = [(doc1, doc2, sim) for doc1, doc2, sim in results if sim >= 0.7]
        if high_sim_pairs:
            print(f"\n{Fore.RED}WARNING: {len(high_sim_pairs)} document pairs with high similarity (>= 0.7) detected{Style.RESET_ALL}")
            for doc1, doc2, sim in high_sim_pairs:
                print(f"  - {Fore.CYAN}{doc1}{Style.RESET_ALL} and {Fore.CYAN}{doc2}{Style.RESET_ALL}: {Fore.RED}{sim:.4f}{Style.RESET_ALL}")
    
    def print_tabular(self, results: Set[Tuple[str, str, float]], 
                     table_format: Optional[str] = None) -> None:
        """
        Print similarity results in a table format.
        
        Args:
            results: Set of tuples with document pairs and similarities
            table_format: Table format for tabulate
        """
        format_to_use = table_format or self.default_table_format
        table_str = ResultFormatter.to_tabulate(results, self.precision, format_to_use)
        print(table_str)
    
    def save_results(self, results: Set[Tuple[str, str, float]], 
                    output_file: str,
                    format: Optional[Union[str, OutputFormat]] = None) -> None:
        """
        Save similarity results to a file in the specified format.
        
        Args:
            results: Set of tuples with document pairs and similarities
            output_file: Path to save the output
            format: Output format to use
        """
        # Determine format from file extension if not specified
        if format is None:
            ext = os.path.splitext(output_file)[1].lower()
            if ext == '.json':
                format = OutputFormat.JSON
            elif ext == '.csv':
                format = OutputFormat.CSV
            elif ext == '.html':
                format = OutputFormat.HTML
            elif ext == '.md':
                format = OutputFormat.MARKDOWN
            elif ext == '.tex':
                format = OutputFormat.LATEX
            else:
                format = self.default_output_format
        
        # Convert string to enum if needed
        if isinstance(format, str):
            try:
                format = OutputFormat(format.lower())
            except ValueError:
                format = self.default_output_format
        
        # Generate the appropriate output
        if format == OutputFormat.JSON:
            output = ResultFormatter.to_json(results, self.precision, True)
        elif format == OutputFormat.CSV:
            output = ResultFormatter.to_csv_str(results, self.precision)
        elif format == OutputFormat.HTML:
            output = ResultFormatter.to_html_table(results, self.precision)
        elif format == OutputFormat.MARKDOWN:
            output = ResultFormatter.to_markdown_table(results, self.precision)
        elif format == OutputFormat.LATEX:
            output = ResultFormatter.to_latex_table(results, self.precision)
        else:
            # Default to JSON if format not recognized
            output = ResultFormatter.to_json(results, self.precision, True)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(output)
            
        print(f"Results saved to {output_file} in {format.value} format.")
    
    def visualize(self, results: Set[Tuple[str, str, float]], 
                 visualization_type: str = "heatmap",
                 output_file: Optional[str] = None,
                 output_format: str = "png",
                 show: bool = True) -> None:
        """
        Visualize similarity results.
        
        Args:
            results: Set of tuples with document pairs and similarities
            visualization_type: Type of visualization ('heatmap' or 'bar')
            output_file: Optional path to save the visualization
            output_format: Format to save the image ('png', 'jpg', 'jpeg', 'svg', 'pdf')
            show: Whether to display the visualization
        """
        # Validate and normalize image format
        valid_formats = ['png', 'jpg', 'jpeg', 'svg', 'pdf', 'eps', 'tiff']
        output_format = output_format.lower()
        
        if output_format not in valid_formats:
            print(f"Unsupported image format: {output_format}. Using png instead.")
            output_format = 'png'
        
        # Adjust output file if specified to use the requested format
        if output_file:
            # If output_file already has an extension, replace it
            base, ext = os.path.splitext(output_file)
            if ext.lower() not in [f'.{fmt}' for fmt in valid_formats]:
                # No valid extension, add the specified format
                output_file = f"{base}.{output_format}"
            elif ext[1:].lower() != output_format:
                # Has extension but different from requested format
                output_file = f"{base}.{output_format}"
        
        # Choose visualization method
        if visualization_type.lower() == "heatmap":
            self.visualizer.plot_similarity_heatmap(
                results, 
                output_file=output_file,
                show=show
            )
        elif visualization_type.lower() == "bar":
            self.visualizer.plot_similarity_bar_chart(
                results, 
                output_file=output_file,
                show=show
            )
        else:
            print(f"Unsupported visualization type: {visualization_type}")
    
    def generate_full_report(self, results: Set[Tuple[str, str, float]], 
                           output_dir: str,
                           include_visualizations: bool = True,
                           report_title: str = "Document Similarity Analysis") -> None:
        """
        Generate a comprehensive report with results in multiple formats.
        
        Args:
            results: Set of tuples with document pairs and similarities
            output_dir: Directory to save report files
            include_visualizations: Whether to include visualizations
            report_title: Title for the report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for file names
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results in various formats
        self.save_results(results, os.path.join(output_dir, f"results_{timestamp}.json"), OutputFormat.JSON)
        self.save_results(results, os.path.join(output_dir, f"results_{timestamp}.csv"), OutputFormat.CSV)
        self.save_results(results, os.path.join(output_dir, f"results_{timestamp}.html"), OutputFormat.HTML)
        self.save_results(results, os.path.join(output_dir, f"results_{timestamp}.md"), OutputFormat.MARKDOWN)
        
        # Generate visualizations if requested
        if include_visualizations:
            self.visualize(results, "heatmap", 
                         output_file=os.path.join(output_dir, f"heatmap_{timestamp}.png"),
                         show=False)
                         
            self.visualize(results, "bar", 
                         output_file=os.path.join(output_dir, f"barchart_{timestamp}.png"),
                         show=False)
        
        # Generate a summary report in plain text
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"{report_title}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Document statistics
            all_docs = set()
            similarities = []
            
            for doc1, doc2, sim in results:
                all_docs.add(doc1)
                all_docs.add(doc2)
                similarities.append(sim)
            
            f.write(f"Total documents: {len(all_docs)}\n")
            f.write(f"Total comparisons: {len(results)}\n")
            f.write(f"Average similarity: {sum(similarities) / len(similarities) if similarities else 0:.4f}\n")
            f.write(f"Maximum similarity: {max(similarities) if similarities else 0:.4f}\n")
            f.write(f"Minimum similarity: {min(similarities) if similarities else 0:.4f}\n\n")
            
            # List all documents
            f.write("Documents analyzed:\n")
            for doc in sorted(all_docs):
                f.write(f"  - {doc}\n")
            f.write("\n")
            
            # Results sorted by similarity
            f.write("Similarity results (sorted by similarity):\n")
            for doc1, doc2, sim in sorted(results, key=lambda x: x[2], reverse=True):
                f.write(f"  {doc1} and {doc2}: {sim:.4f} similarity ({sim*100:.2f}%)\n")
        
        print(f"Comprehensive report generated in {output_dir}")
