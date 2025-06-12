import param

from lumen.ai.analysis import Analysis

from .views import ManifoldMapPanel, UMAPPanel


class ManifoldMapAnalysis(Analysis):
    """
    Use this to visualize any requests for UMAP, PCA, tSNE results,
    unless explicitly otherwise specified by the user.
    """
    def __call__(self, pipeline):
        return ManifoldMapPanel(pipeline=pipeline)


class UMAPAnalysis(Analysis):
    category = param.Selector(
        objects=[
            "cell_type",
            "sample_name",
            "Procedure_Type",
            "n_genes_by_counts",
            "total_counts",
            "total_counts_mt",
            "pct_counts_mt",
            "total_counts_ribo",
            "pct_counts_ribo",
            "Phenograph_cluster",
            "histology",
            "sample_number",
            "hta_id",
            "Gender",
            "Ethnicity",
            "Race",
            "Smoking Status",
            "Pack Years",
            "Stage at Dx",
            "Tissue Type",
            "ProcedureType",
            "Treatment Status",
            "Tissue Site",
            "hta_donor_id",
            "cell_lineage",
            "assay_ontology_term_id",
            "sex_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "cell_type_ontology_term_id",
            "donor_id",
            "development_stage_ontology_term_id",
            "is_primary_data",
            "organism_ontology_term_id",
            "disease_ontology_term_id",
            "suspension_type",
            "tissue_ontology_term_id",
            "tissue_type",
            "assay",
            "disease",
            "organism",
            "sex",
            "tissue",
            "self_reported_ethnicity",
            "development_stage",
            "observation_joinid",
        ]
    )

    def __call__(self, pipeline):
        return UMAPPanel(pipeline=pipeline)



class LeidenUMAPAnalysis(UMAPAnalysis):
    """
    Applies Leiden clustering to the UMAP embedding of the data.
    """

    category = param.Selector(default="leiden", objects=["leiden"])

    resolution = param.Number(default=0.5, bounds=(0.1, 2.0), doc="Resolution parameter for Leiden clustering.")

    def __call__(self, pipeline):
        return UMAPPanel(
            pipeline=pipeline,
            category=self.category,
            operation="Leiden",
            operation_kwargs={"resolution": self.resolution},
        )

class ComputeEmbeddingAnalysis(UMAPAnalysis):
    """
    Applies a generic embedding computation operation to the data.
    """

    def __call__(self, pipeline):
        return UMAPPanel(
            pipeline=pipeline,
            operation="ComputeEmbedding",
        )
