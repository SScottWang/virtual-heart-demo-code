"""
Geneformer Utilities for Spatial Transcriptomics Analysis
==========================================================

This module provides reusable functions for:
1. Data preprocessing and tokenization
2. Cell embedding extraction
3. In silico perturbation analysis
4. Visualization and statistical analysis

Author: Shuguang Wang
Date: 2025-12-24
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import mygene
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

from geneformer import TranscriptomeTokenizer, EmbExtractor, InSilicoPerturber, InSilicoPerturberStats


# =====================================================================
# PART 1: Data Preprocessing and Tokenization
# =====================================================================

class DataPreprocessor:
    """Handle data preprocessing for Geneformer"""
    
    def __init__(self, mgi_file_path: str = "/home/wsg/SSW/data/HOM_MouseHumanSequence.rpt"):
        """
        Initialize the preprocessor.
        
        Args:
            mgi_file_path: Path to MGI ortholog mapping file
        """
        self.mgi_file_path = mgi_file_path
        self.mg = mygene.MyGeneInfo()
        
    def add_qc_metrics(self, adata: sc.AnnData, count_layer: Optional[str] = None) -> sc.AnnData:
        """
        Add QC metrics required by Geneformer.
        
        Args:
            adata: AnnData object
            count_layer: Name of layer containing raw counts (if None, uses adata.X)
            
        Returns:
            AnnData with QC metrics added
        """
        print("Adding QC metrics...")
        
        # Move raw counts to X if needed
        if count_layer is not None:
            print(f"Moving counts from layer '{count_layer}' to adata.X...")
            adata.X = adata.layers[count_layer].copy()
        
        # Calculate QC metrics
        print("Calculating n_counts and n_genes...")
        sc.pp.calculate_qc_metrics(
            adata,
            expr_type='counts',
            var_type='genes',
            percent_top=None,
            log1p=False,
            inplace=True
        )
        
        # Add required columns
        adata.obs['n_counts'] = adata.obs['total_counts']
        adata.obs['filter_pass'] = 1
        
        print(f"✅ QC metrics added. n_counts range: [{adata.obs['n_counts'].min():.0f}, {adata.obs['n_counts'].max():.0f}]")
        
        return adata
    
    def convert_mouse_to_human_ensembl(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Convert mouse gene symbols to human Ensembl IDs.
        
        Args:
            adata: AnnData with mouse gene symbols as var_names
            
        Returns:
            AnnData with human Ensembl IDs as var_names
        """
        print("🚀 Converting Mouse Symbol -> Human Ensembl ID...")
        
        mouse_symbols = adata.var_names.tolist()
        print(f"Input: {len(mouse_symbols)} mouse genes")
        
        # Step 1: Parse MGI file
        print("Step 1: Parsing MGI ortholog file...")
        mgi_df = pd.read_csv(self.mgi_file_path, sep='\t')
        
        mouse_rows = mgi_df[mgi_df['Common Organism Name'] == 'mouse, laboratory']
        human_rows = mgi_df[mgi_df['Common Organism Name'] == 'human']
        
        mouse_rows = mouse_rows[['DB Class Key', 'Symbol']].rename(columns={'Symbol': 'Mouse_Symbol'})
        human_rows = human_rows[['DB Class Key', 'Symbol']].rename(columns={'Symbol': 'Human_Symbol'})
        
        ortho_df = pd.merge(mouse_rows, human_rows, on='DB Class Key', how='inner')
        mouse_to_human = dict(zip(ortho_df['Mouse_Symbol'], ortho_df['Human_Symbol']))
        
        print(f"✅ Found {len(mouse_to_human)} ortholog pairs")
        
        # Step 2: Convert Human Symbol to Ensembl ID
        print("Step 2: Converting Human Symbol -> Ensembl ID...")
        needed_human_symbols = [mouse_to_human[m] for m in mouse_symbols if m in mouse_to_human]
        
        res_human = self.mg.querymany(
            needed_human_symbols,
            scopes='symbol',
            fields='ensembl.gene',
            species='human'
        )
        df_human_id = pd.DataFrame(res_human)
        
        def get_human_id(x):
            if isinstance(x, dict):
                return x.get('gene')
            elif isinstance(x, list):
                return x[0].get('gene') if len(x) > 0 else None
            return None
        
        df_human_id['human_ensg'] = df_human_id['ensembl'].apply(get_human_id)
        df_human_id = df_human_id.dropna(subset=['human_ensg']).drop_duplicates(subset=['query'])
        human_to_ensg = dict(zip(df_human_id['query'], df_human_id['human_ensg']))
        
        # Step 3: Create final mapping
        print("Step 3: Applying mapping...")
        final_map = {}
        for msym in mouse_symbols:
            if msym in mouse_to_human:
                hsym = mouse_to_human[msym]
                if hsym in human_to_ensg:
                    final_map[msym] = human_to_ensg[hsym]
        
        adata.var['human_ensembl_id'] = adata.var_names.map(final_map)
        mapped_count = adata.var['human_ensembl_id'].notna().sum()
        print(f"🎉 Result: {mapped_count} / {adata.n_vars} genes successfully mapped")
        
        # Filter and update
        adata = adata[:, adata.var['human_ensembl_id'].notna()].copy()
        adata.var['mouse_symbol'] = adata.var_names
        adata.var['ensembl_id'] = adata.var['human_ensembl_id']
        adata.var_names = adata.var['ensembl_id']
        del adata.var['human_ensembl_id']
        adata.var.index.name = None
        adata.var_names_make_unique()
        
        print(f"✅ Ready! Gene IDs example: {list(adata.var_names[:5])}")
        
        return adata


def tokenize_data(
    input_dir: str,
    output_dir: str,
    output_prefix: str,
    custom_attr_dict: Optional[Dict[str, str]] = None,
    file_format: str = "h5ad",
    nproc: int = 16
) -> None:
    """
    Tokenize preprocessed data for Geneformer.
    
    Args:
        input_dir: Directory containing h5ad/loom files
        output_dir: Output directory for tokenized data
        output_prefix: Prefix for output files
        custom_attr_dict: Custom attributes to preserve (e.g., {"cell_type": "cell_type"})
        file_format: Input file format ("h5ad" or "loom")
        nproc: Number of processes
    """
    print(f"🚀 Tokenizing data from {input_dir}...")
    
    if custom_attr_dict is None:
        custom_attr_dict = {}
    
    tk = TranscriptomeTokenizer(custom_attr_dict, nproc=nproc)
    tk.tokenize_data(input_dir, output_dir, output_prefix, file_format=file_format)
    
    print(f"✅ Tokenization complete! Output: {output_dir}/{output_prefix}.dataset")


# =====================================================================
# PART 2: Embedding Extraction
# =====================================================================

def extract_embeddings(
    model_path: str,
    token_data_path: str,
    output_dir: str,
    output_prefix: str,
    emb_labels: List[str] = ["cell_type"],
    labels_to_plot: List[str] = ["cell_type"],
    forward_batch_size: int = 50,
    gpu_id: Optional[int] = None,
    nproc: int = 16
) -> pd.DataFrame:
    """
    Extract cell embeddings from tokenized data.
    
    Args:
        model_path: Path to Geneformer model
        token_data_path: Path to tokenized dataset
        output_dir: Output directory
        output_prefix: Prefix for output files
        emb_labels: Labels to include in embedding output
        labels_to_plot: Labels to use for plotting
        forward_batch_size: Batch size for forward pass
        gpu_id: GPU device ID (None for auto)
        nproc: Number of processes
        
    Returns:
        DataFrame containing embeddings and metadata
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🎮 Using GPU {gpu_id}")
    
    print("🚀 Extracting embeddings...")
    
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        filter_data=None,
        max_ncells=None,
        emb_layer=-1,
        emb_label=emb_labels,
        labels_to_plot=labels_to_plot,
        forward_batch_size=forward_batch_size,
        model_version="V2",
        nproc=nproc
    )
    
    embs = embex.extract_embs(
        model_path,
        token_data_path,
        output_dir,
        output_prefix
    )
    
    print(f"✅ Embeddings extracted! Shape: {embs.shape}")
    
    return embs


# =====================================================================
# PART 3: In Silico Perturbation Analysis
# =====================================================================

class PerturbationAnalyzer:
    """
    Comprehensive in silico perturbation analysis tool.
    """
    
    def __init__(
        self,
        model_path: str,
        token_data_path: str,
        output_dir: str,
        gpu_id: Optional[int] = None,
        forward_batch_size: int = 50,
        nproc: int = 16
    ):
        """
        Initialize perturbation analyzer.
        
        Args:
            model_path: Path to Geneformer model
            token_data_path: Path to tokenized dataset
            output_dir: Output directory for results
            gpu_id: GPU device ID
            forward_batch_size: Batch size
            nproc: Number of processes
        """
        self.model_path = model_path
        self.token_data_path = token_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"🎮 Using GPU {gpu_id}")
    
    def perturb_gene(
        self,
        gene_ensembl_id: str,
        gene_name: str,
        perturb_type: str = "delete"
    ) -> pd.DataFrame:
        """
        Perform in silico perturbation for a single gene.
        
        Args:
            gene_ensembl_id: Human Ensembl ID of the gene (e.g., "ENSG00000183072")
            gene_name: Gene symbol for reference (e.g., "Nkx2-5")
            perturb_type: Type of perturbation ("delete" or "overexpress")
            
        Returns:
            DataFrame with perturbation statistics
        """
        print(f"\n{'='*60}")
        print(f"🧬 Perturbing gene: {gene_name} ({gene_ensembl_id})")
        print(f"{'='*60}\n")
        
        # Step 1: Run perturbation
        print("Step 1: Running in silico perturbation...")
        isp = InSilicoPerturber(
            perturb_type=perturb_type,
            perturb_rank_shift=None,
            genes_to_perturb=[gene_ensembl_id],
            combos=0,
            anchor_gene=None,
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cls",
            cell_emb_style="mean_pool",
            filter_data=None,
            max_ncells=None,
            forward_batch_size=self.forward_batch_size,
            nproc=self.nproc
        )
        
        perturb_output_dir = self.output_dir / f"{gene_name}_perturbation"
        perturb_output_dir.mkdir(exist_ok=True)
        
        isp.perturb_data(
            self.model_path,
            self.token_data_path,
            str(perturb_output_dir),
            f"{gene_name}_perturb"
        )
        
        print("✅ Perturbation complete!")
        
        # Step 2: Calculate statistics
        print("\nStep 2: Calculating statistics...")
        ispstats = InSilicoPerturberStats(
            mode="aggregate_data",
            genes_perturbed=[gene_ensembl_id],
            combos=0,
            anchor_gene=None,
            cell_states_to_model=None,
            model_version="V2"
        )
        
        stats_output_dir = self.output_dir / f"{gene_name}_stats"
        stats_output_dir.mkdir(exist_ok=True)
        
        ispstats.get_stats(
            str(perturb_output_dir),
            None,
            str(stats_output_dir),
            f"{gene_name}_stats"
        )
        
        # Load results
        stats_file = stats_output_dir / f"{gene_name}_stats.csv"
        df_stats = pd.read_csv(stats_file)
        
        print(f"✅ Statistics saved to: {stats_file}")
        print(f"   Cells analyzed: {len(df_stats)}")
        
        return df_stats
    
    def map_impact_to_adata(
        self,
        adata: sc.AnnData,
        stats_df: pd.DataFrame,
        gene_name: str,
        gene_ensembl_id: str
    ) -> sc.AnnData:
        """
        Map perturbation impact scores back to AnnData object.
        
        Args:
            adata: Original AnnData object with expression data
            stats_df: Statistics DataFrame from perturbation
            gene_name: Gene symbol (for column naming)
            gene_ensembl_id: Ensembl ID to match expression
            
        Returns:
            AnnData with impact scores added to obs
        """
        print(f"\n📍 Mapping {gene_name} impact to spatial coordinates...")
        
        # Find cosine similarity column
        cosine_col = [c for c in stats_df.columns if "cosine" in c.lower() or "shift" in c.lower()][0]
        impact_scores = 1 - stats_df[cosine_col].values
        
        # Find cells expressing the gene
        if gene_ensembl_id in adata.var_names:
            expr_data = adata[:, gene_ensembl_id].X
            if hasattr(expr_data, "toarray"):
                expr_data = expr_data.toarray().flatten()
            else:
                expr_data = expr_data.flatten()
        else:
            # Try to find by mouse symbol
            mouse_symbol = gene_name
            if 'mouse_symbol' in adata.var.columns:
                matching_genes = adata.var[adata.var['mouse_symbol'] == mouse_symbol]
                if len(matching_genes) > 0:
                    gene_ensembl_id = matching_genes.index[0]
                    expr_data = adata[:, gene_ensembl_id].X
                    if hasattr(expr_data, "toarray"):
                        expr_data = expr_data.toarray().flatten()
                    else:
                        expr_data = expr_data.flatten()
                else:
                    raise ValueError(f"Gene {gene_name} / {gene_ensembl_id} not found in adata")
            else:
                raise ValueError(f"Gene {gene_name} / {gene_ensembl_id} not found in adata")
        
        # Create full impact vector
        total_impact = np.zeros(adata.n_obs)
        is_positive = expr_data > 0
        
        # Align and assign
        if np.sum(is_positive) == len(impact_scores):
            total_impact[is_positive] = impact_scores
            print(f"✅ Perfect match! {np.sum(is_positive)} expressing cells")
        elif abs(np.sum(is_positive) - len(impact_scores)) < 100:
            print(f"⚠️  Slight mismatch (AnnData: {np.sum(is_positive)} vs Stats: {len(impact_scores)})")
            true_indices = np.where(is_positive)[0]
            valid_count = min(len(true_indices), len(impact_scores))
            total_impact[true_indices[:valid_count]] = impact_scores[:valid_count]
        else:
            raise ValueError(f"❌ Large mismatch: {np.sum(is_positive)} expressing cells vs {len(impact_scores)} impact scores")
        
        # Add to adata
        impact_col_name = f"{gene_name}_Impact"
        adata.obs[impact_col_name] = total_impact
        
        print(f"✅ Impact mapped! Column: '{impact_col_name}'")
        print(f"   Impact range: [{total_impact[total_impact>0].min():.6f}, {total_impact[total_impact>0].max():.6f}]")
        
        return adata


# =====================================================================
# PART 4: Visualization Functions
# =====================================================================

def plot_3d_spatial(
    adata: sc.AnnData,
    spatial_key: str = 'aligned_spatial_3D_new',
    color_by: str = 'cell_type',
    title: str = None,
    save_path: Optional[str] = None,
    point_size: float = 1.5,
    show_title: bool = False  # Control whether to show title
) -> go.Figure:
    """
    Create interactive 3D spatial plot for documentation embedding.
    
    Args:
        adata: AnnData object
        spatial_key: Key in obsm containing 3D coordinates
        color_by: Column in obs to color by
        title: Plot title (default: None)
        save_path: Path to save HTML (optional)
        point_size: Size of scatter points (default: 1.5)
        show_title: Whether to show title (default: False)
        
    Returns:
        Plotly Figure object
    """
    df_plot = pd.DataFrame(
        adata.obsm[spatial_key],
        columns=['x', 'y', 'z']
    )
    df_plot['color'] = adata.obs[color_by].values
    
    fig = px.scatter_3d(
        df_plot,
        x='x', y='y', z='z',
        color='color',
        opacity=0.7,
        title=title if show_title else None
    )
    
    # Update marker size
    fig.update_traces(marker=dict(size=point_size))
    
    # Update legend settings - enlarge markers and customize title
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=10 if not show_title else 30),
        legend=dict(
            title=dict(
                text="Heart Region",
                font=dict(size=12, family="Arial, sans-serif")
            ),
            font=dict(size=11),
            itemsizing='constant',
            tracegroupgap=5,
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle'
        ),
        # Adjust figure size for document embedding
        height=500,
        width=650
    )
    
    # Enlarge markers in legend
    fig.for_each_trace(lambda t: t.update(marker=dict(size=point_size, line=dict(width=0))))
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Plot saved to: {save_path}")
    
    return fig


def plot_perturbation_impact_3d(
    adata: sc.AnnData,
    gene_name: str,
    spatial_key: str = 'aligned_spatial_3D_new',
    region_key: Optional[str] = None,
    quantile_clip: Tuple[float, float] = (0.05, 0.95),
    save_path: Optional[str] = None,
    point_size: float = 1.5  # Size of scatter points
) -> go.Figure:
    """
    Visualize perturbation impact in 3D space.
    
    Args:
        adata: AnnData object with impact scores
        gene_name: Gene symbol (column will be '{gene_name}_Impact')
        spatial_key: Key in obsm for 3D coordinates
        region_key: Optional key for anatomical regions
        quantile_clip: Quantiles for color scale clipping
        save_path: Path to save HTML
        point_size: Size of scatter points (default: 1.5)
        
    Returns:
        Plotly Figure object
    """
    impact_col = f"{gene_name}_Impact"
    
    if impact_col not in adata.obs.columns:
        raise ValueError(f"Impact column '{impact_col}' not found in adata.obs")
    
    df_plot = pd.DataFrame(
        adata.obsm[spatial_key],
        columns=['x', 'y', 'z']
    )
    df_plot['Impact'] = adata.obs[impact_col].values
    
    if region_key and region_key in adata.obs.columns:
        df_plot['Region'] = adata.obs[region_key].values
        hover_data = ['Region']
    else:
        hover_data = None
    
    # Clip color range
    impact_positive = df_plot['Impact'][df_plot['Impact'] > 0]
    if len(impact_positive) > 0:
        vmin = np.quantile(impact_positive, quantile_clip[0])
        vmax = np.quantile(impact_positive, quantile_clip[1])
    else:
        vmin, vmax = 0, 1
    
    fig = px.scatter_3d(
        df_plot,
        x='x', y='y', z='z',
        color='Impact',
        color_continuous_scale='Reds',
        range_color=[vmin, vmax],
        opacity=0.7,  # 修改：降低透明度
        title=f"{gene_name} Deletion Impact",
        hover_data=hover_data
    )
    
    # 修改：更新marker大小
    fig.update_traces(marker=dict(size=point_size))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Plot saved to: {save_path}")
    
    return fig


def plot_perturbation_comparison_3d(
    adata: sc.AnnData,
    gene_name: str,
    gene_ensembl_id: str,
    spatial_key: str = 'aligned_spatial_3D_new',
    region_key: Optional[str] = None,
    save_path: Optional[str] = None,
    point_size: float = 1.5
) -> go.Figure:
    """
    Two-way comparison with linked selection: Cell type and KO Impact.
    Optimized for documentation embedding.
    
    Args:
        adata: AnnData object
        gene_name: Gene symbol
        gene_ensembl_id: Ensembl ID for expression data
        spatial_key: Key for 3D coordinates
        region_key: Key for cell type/region annotations
        save_path: Path to save HTML
        point_size: Size of scatter points (default: 1.5)
        
    Returns:
        Plotly Figure object
    """
    impact_col = f"{gene_name}_Impact"
    
    # Get impact
    impact = adata.obs[impact_col].values
    
    # Prepare data
    df_plot = pd.DataFrame(
        adata.obsm[spatial_key],
        columns=['x', 'y', 'z']
    )
    
    # Create 2 subplots (Cell Type + Impact) with no titles
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=None,
        horizontal_spacing=0.02  # Minimize horizontal spacing
    )
    
    # Left: Cell Type with legend
    if region_key and region_key in adata.obs.columns:
        unique_types = sorted(adata.obs[region_key].unique())
        colors = px.colors.qualitative.Set3[:len(unique_types)]
        color_map = {t: colors[i % len(colors)] for i, t in enumerate(unique_types)}
        
        for cell_type in unique_types:
            mask = adata.obs[region_key] == cell_type
            indices = np.where(mask)[0]
            
            fig.add_trace(
                go.Scatter3d(
                    x=df_plot.loc[indices, 'x'],
                    y=df_plot.loc[indices, 'y'],
                    z=df_plot.loc[indices, 'z'],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=color_map[cell_type],
                        opacity=0.6,
                        line=dict(width=0)
                    ),
                    name=cell_type,
                    legendgroup=cell_type,
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Right: Impact (grouped by cell type for linked interaction)
    if region_key and region_key in adata.obs.columns:
        i_max = np.quantile(impact[impact > 0], 0.95) if np.sum(impact > 0) > 0 else 1
        
        for cell_type in unique_types:
            mask = adata.obs[region_key] == cell_type
            indices = np.where(mask)[0]
            
            fig.add_trace(
                go.Scatter3d(
                    x=df_plot.loc[indices, 'x'],
                    y=df_plot.loc[indices, 'y'],
                    z=df_plot.loc[indices, 'z'],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=impact[indices],
                        colorscale='Reds',
                        cmin=0,
                        cmax=i_max,
                        opacity=0.7,
                        showscale=True,
                        colorbar=dict(
                            x=0.98,
                            len=0.5,
                            thickness=12,
                            title=dict(
                                text="Impact",
                                side="right",
                                font=dict(size=10)
                            ),
                            tickfont=dict(size=9)
                        ),
                        line=dict(width=0)
                    ),
                    name=cell_type,
                    legendgroup=cell_type,
                    showlegend=False
                ),
                row=1, col=2
            )
    else:
        # If no region_key, simply display impact
        i_max = np.quantile(impact[impact > 0], 0.95) if np.sum(impact > 0) > 0 else 1
        fig.add_trace(
            go.Scatter3d(
                x=df_plot['x'], y=df_plot['y'], z=df_plot['z'],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=impact,
                    colorscale='Reds',
                    cmin=0,
                    cmax=i_max,
                    opacity=0.7,
                    colorbar=dict(
                        x=0.98,
                        len=0.5,
                        thickness=12,
                        tickfont=dict(size=9)
                    )
                ),
                name='Impact',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Adjust layout for document embedding, no title
    fig.update_layout(
        showlegend=True,
        height=450,
        width=850,  # Further reduce width for complete display
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            title=dict(
                text="Heart Region",
                font=dict(size=10, family="Arial, sans-serif")
            ),
            font=dict(size=9),
            itemsizing='constant',
            tracegroupgap=2,
            itemwidth=30,  # Plotly minimum value is 30
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.85)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    # Update scene settings
    fig.update_scenes(
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)  # Adjust initial view angle
        )
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✅ Plot saved to: {save_path}")
        print(f"💡 Tip: Click legend items to show/hide cell types in both plots")
    
    return fig


def plot_regional_impact_boxplot(
    adata: sc.AnnData,
    gene_name: str,
    region_key: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)  # Moderate figure size for document embedding
) -> plt.Figure:
    """
    Create boxplot showing regional specificity of perturbation impact.
    
    Args:
        adata: AnnData object
        gene_name: Gene symbol
        region_key: Column in obs containing region labels
        save_path: Path to save PNG (not PDF)
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    impact_col = f"{gene_name}_Impact"
    
    df_eval = pd.DataFrame({
        'Region': adata.obs[region_key],
        'Impact': adata.obs[impact_col].values
    })
    
    # Calculate regional statistics
    region_stats = df_eval.groupby('Region')['Impact'].agg(['count', 'mean', 'median', 'std', 'max'])
    region_stats = region_stats.sort_values(by='mean', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"🏆 Regional {gene_name} Dependency Ranking (Mean Impact)")
    print(f"{'='*60}")
    print(region_stats)
    
    # Calculate appropriate y-axis range using aggressive strategy to ensure bars are visible
    all_impacts = df_eval['Impact'].values
    
    # Use multiple statistical methods
    mean_impact = df_eval['Impact'].mean()
    median_impact = df_eval['Impact'].median()
    q75 = df_eval['Impact'].quantile(0.75)
    q95 = df_eval['Impact'].quantile(0.95)
    max_impact = df_eval['Impact'].max()
    
    print(f"\n📊 Impact Statistics:")
    print(f"  Mean: {mean_impact:.8f}")
    print(f"  Median: {median_impact:.8f}")
    print(f"  75th percentile: {q75:.8f}")
    print(f"  95th percentile: {q95:.8f}")
    print(f"  Max: {max_impact:.8f}")
    
    # Use 3x mean or 1.5x 95th percentile, whichever is larger
    y_max = max(mean_impact * 3, q95 * 1.5, max_impact * 1.1)
    
    # If still too small, use 1.2x max
    if y_max < max_impact:
        y_max = max_impact * 1.2
    
    # Ensure a reasonable minimum range
    if y_max < 1e-8:
        y_max = 1e-6
    
    y_min = 0
    
    print(f"  Y-axis range: [0, {y_max:.8f}]")
    
    # Create plot with moderate size
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("white")
    
    sns.boxplot(
        data=df_eval,
        x='Region',
        y='Impact',
        order=region_stats.index,
        palette="viridis",
        showfliers=False,
        ax=ax
    )
    
    # Set y-axis range
    ax.set_ylim(y_min, y_max)
    
    # Use scientific notation for y-axis
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Font settings for embedded display
    ax.set_ylabel("Impact (1 - Cosine Similarity)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Heart Region", fontsize=11, fontweight='bold')
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xticks(rotation=45, ha='right')
    
    # Grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        # Save as PNG format with moderate DPI
        png_path = save_path.replace('.pdf', '.png') if save_path.endswith('.pdf') else save_path
        fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Boxplot saved to: {png_path}")
    
    return fig


# =====================================================================
# PART 5: High-Level Workflow Functions
# =====================================================================

def complete_perturbation_workflow(
    adata: sc.AnnData,
    gene_name: str,
    gene_ensembl_id: str,
    model_path: str,
    token_data_path: str,
    output_base_dir: str,
    spatial_key: str = 'aligned_spatial_3D_new',
    region_key: Optional[str] = None,
    gpu_id: Optional[int] = None,
    create_plots: bool = True
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """
    Complete end-to-end perturbation analysis workflow.
    
    Args:
        adata: AnnData object with spatial coordinates
        gene_name: Gene symbol (e.g., "Nkx2-5")
        gene_ensembl_id: Human Ensembl ID (e.g., "ENSG00000183072")
        model_path: Path to Geneformer model
        token_data_path: Path to tokenized data
        output_base_dir: Base directory for outputs
        spatial_key: Key in obsm for spatial coordinates
        region_key: Optional key for anatomical regions
        gpu_id: GPU device ID
        create_plots: Whether to generate visualization plots
        
    Returns:
        Tuple of (updated AnnData, statistics DataFrame)
    """
    print("\n" + "="*70)
    print(f"🚀 COMPLETE PERTURBATION WORKFLOW: {gene_name}")
    print("="*70 + "\n")
    
    output_dir = Path(output_base_dir) / gene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Run perturbation
    analyzer = PerturbationAnalyzer(
        model_path=model_path,
        token_data_path=token_data_path,
        output_dir=str(output_dir),
        gpu_id=gpu_id
    )
    
    stats_df = analyzer.perturb_gene(
        gene_ensembl_id=gene_ensembl_id,
        gene_name=gene_name,
        perturb_type="delete"
    )
    
    # Step 2: Map to spatial data
    adata = analyzer.map_impact_to_adata(
        adata=adata,
        stats_df=stats_df,
        gene_name=gene_name,
        gene_ensembl_id=gene_ensembl_id
    )
    
    # Step 3: Generate visualizations
    if create_plots:
        print("\n📊 Generating visualizations...")
        
        # 3D impact plot
        fig1 = plot_perturbation_impact_3d(
            adata=adata,
            gene_name=gene_name,
            spatial_key=spatial_key,
            region_key=region_key,
            save_path=str(output_dir / f"{gene_name}_impact_3d.html")
        )
        
        # Expression vs Impact comparison (with cell type)
        fig2 = plot_perturbation_comparison_3d(
            adata=adata,
            gene_name=gene_name,
            gene_ensembl_id=gene_ensembl_id,
            spatial_key=spatial_key,
            region_key=region_key,  # Pass region_key for cell type display
            save_path=str(output_dir / f"{gene_name}_comparison_3d.html")
        )
        
        # Regional boxplot (if region info available)
        if region_key and region_key in adata.obs.columns:
            fig3 = plot_regional_impact_boxplot(
                adata=adata,
                gene_name=gene_name,
                region_key=region_key,
                save_path=str(output_dir / f"{gene_name}_regional_boxplot.png")
            )
            plt.close(fig3)
    
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETE!")
    print(f"📁 All results saved to: {output_dir}")
    print("="*70 + "\n")
    
    return adata, stats_df


def batch_perturbation_analysis(
    adata: sc.AnnData,
    genes_to_test: List[Tuple[str, str]],  # List of (gene_name, ensembl_id) tuples
    model_path: str,
    token_data_path: str,
    output_base_dir: str,
    spatial_key: str = 'aligned_spatial_3D_new',
    region_key: Optional[str] = None,
    gpu_id: Optional[int] = None
) -> Dict[str, Tuple[sc.AnnData, pd.DataFrame]]:
    """
    Run perturbation analysis for multiple genes.
    
    Args:
        adata: AnnData object
        genes_to_test: List of (gene_name, ensembl_id) tuples
        model_path: Path to Geneformer model
        token_data_path: Path to tokenized data
        output_base_dir: Base output directory
        spatial_key: Key for spatial coordinates
        region_key: Key for regions
        gpu_id: GPU device ID
        
    Returns:
        Dictionary mapping gene names to (AnnData, stats_df) tuples
    """
    results = {}
    
    print("\n" + "="*70)
    print(f"🚀 BATCH PERTURBATION ANALYSIS: {len(genes_to_test)} genes")
    print("="*70 + "\n")
    
    for i, (gene_name, gene_ensembl_id) in enumerate(genes_to_test, 1):
        print(f"\n[{i}/{len(genes_to_test)}] Processing {gene_name}...")
        
        try:
            adata_updated, stats_df = complete_perturbation_workflow(
                adata=adata.copy(),  # Use copy to avoid conflicts
                gene_name=gene_name,
                gene_ensembl_id=gene_ensembl_id,
                model_path=model_path,
                token_data_path=token_data_path,
                output_base_dir=output_base_dir,
                spatial_key=spatial_key,
                region_key=region_key,
                gpu_id=gpu_id,
                create_plots=True
            )
            
            results[gene_name] = (adata_updated, stats_df)
            print(f"✅ {gene_name} complete!")
            
        except Exception as e:
            print(f"❌ Error processing {gene_name}: {str(e)}")
            continue
    
    print("\n" + "="*70)
    print(f"✅ BATCH ANALYSIS COMPLETE! {len(results)}/{len(genes_to_test)} genes successful")
    print("="*70 + "\n")
    
    return results

