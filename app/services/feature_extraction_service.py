# -*- coding: utf-8 -*-
import logging
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox
from app.views.pca_explorer_dialog import PCAExplorerDialog
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureExtractionService:
    """
    Service to handle feature extraction logic, including the dual-mode PCA.
    """
    def __init__(self, data_model, logger, parent_view=None):
        """
        Initializes the service.

        Args:
            data_model: The application's data model.
            logger: The logger instance for logging messages.
            parent_view: The parent widget, used for displaying dialogs.
        """
        self.data_model = data_model
        self.logger = logger if logger else logging.getLogger(__name__)
        self.parent_view = parent_view

    def run_feature_extraction(self, method: str, params: Dict[str, Any]):
        """
        Routes the feature extraction task to the appropriate method.
        """
        if "PCA" in method:
            self._run_pca_workflow(params)
        else:
            self.logger.warning(f"Feature extraction method '{method}' is not yet implemented in the service.")
            QMessageBox.warning(self.parent_view, "Not Implemented", f"Feature extraction method '{method}' is not yet implemented in the service.")

    def _run_pca_workflow(self, params: Dict[str, Any]):
        """
        Manages the PCA workflow, deciding between exploratory and modeling modes.
        """
        try:
            self.logger.info(f"Attempting to apply PCA with params: {params}")
            
            # This service only handles the exploratory case for now.
            # The modeling case is handled by the controller.
            if not self.data_model.has_partitioned_data():
                self.logger.info("Data is not partitioned. Launching PCA Explorer.")
                X = self.data_model.get_X()
                y = self.data_model.get_y()

                if X is None or X.empty:
                    QMessageBox.warning(self.parent_view, "Data Error", "No data available for PCA.")
                    return
                
                # V1.4.0: Direct PCA implementation (plugin removed)
                n_components = params.get('n_components', 10)
                
                # Standardize data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Perform PCA
                pca = PCA(n_components=n_components)
                scores = pca.fit_transform(X_scaled)
                
                # Calculate cumulative explained variance
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                
                # Build results dictionary
                # V1.4.1: Add 'model' key and 'cumulative_explained_variance' for compatibility with dialog
                pca_results = {
                    'model': pca,  # PCA model object
                    'scores': scores,
                    'loadings': pca.components_.T,
                    'explained_variance': pca.explained_variance_,
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'cumulative_explained_variance': cumulative_variance,  # V1.4.1: Add cumulative variance
                    'n_components': n_components
                }
                
                if pca_results:
                    self._launch_explorer_dialog(pca_results, self.data_model.get_wavelengths(), y)
                else:
                    QMessageBox.critical(self.parent_view, "PCA Error", "Failed to perform exploratory PCA.")
            else:
                self.logger.info("Data is partitioned. This should be handled by the FeatureSelectionController.")
                QMessageBox.information(self.parent_view, "Information", "PCA on partitioned data is part of the modeling workflow and is handled by the existing controller.")

        except Exception as e:
            self.logger.error(f"An error occurred during the PCA workflow: {e}", exc_info=True)
            QMessageBox.critical(self.parent_view, "PCA Workflow Error", f"An unexpected error occurred: {e}")

    def _launch_explorer_dialog(self, pca_results: Dict[str, Any], wavelengths: pd.Series, y: pd.Series):
        """
        Initializes and shows the PCAExplorerDialog with the results.
        """
        if not self.parent_view:
            self.logger.error("Cannot launch PCA dialog without a parent view.")
            return

        try:    
            dialog = PCAExplorerDialog(parent=self.parent_view)
            
            # The plugin returns loadings as a DataFrame (n_features, n_components).
            # The dialog expects a numpy array (n_components, n_features).
            # We get it directly from the PCA model object.
            loadings_for_dialog = pca_results['model'].components_

            # V1.4.1: Handle scores conversion (may be numpy array or DataFrame)
            scores_data = pca_results['scores']
            if hasattr(scores_data, 'to_numpy'):
                scores_array = scores_data.to_numpy()
            else:
                scores_array = scores_data
            
            dialog.set_data(
                pca_instance=pca_results['model'],
                loadings=loadings_for_dialog,
                explained_variance_ratio=pca_results['explained_variance_ratio'],
                cumulative_variance=pca_results['cumulative_explained_variance'],
                X_scores=scores_array,
                wavelengths=wavelengths,
                y=y.to_numpy() if y is not None else None,
                is_modeling_mode=False
            )
            dialog.exec_()
        except Exception as e:
            self.logger.error(f"Failed to launch PCA explorer dialog: {e}", exc_info=True)
            QMessageBox.critical(self.parent_view, "Dialog Error", f"Could not display PCA results: {e}")