# from utils.visualization_window import VisualizationWindow  # Temporarily disabled for packaging
from app.utils.logger import setup_logger
from app.utils.exceptions import ModelingError
from app.services.modeling_service import ModelingService
from app.models.modeling_model import ModelingModel
from app.utils.label_processor import EnhancedLabelProcessor
from app.services.onnx_service import ONNXService
import numpy as np
import pandas as pd
import os

class ModelingController:
    def __init__(self, view, translator, data_model, plugins):
        self.view = view
        self.translator = translator
        self.data_model = data_model
        self.modeling_service = ModelingService(plugins)
        # V1.4.0: Let service access view for dynamically added plugins
        self.modeling_service.view = view
        self.modeling_model = ModelingModel()
        self.onnx_service = ONNXService(save_directory="models")
        self.logger = setup_logger()
        
        from app.services.evaluation_service import EvaluationService
        self.evaluation_service = EvaluationService()
        
        self.modeling_service.evaluation_service = self.evaluation_service
        

        self.label_processor = EnhancedLabelProcessor()
        print("🔧 ModelingController initialized with EnhancedLabelProcessor")

    def train_model(self, algorithm, params):
        try:
            if not self.data_model.has_partitioned_data():
                raise ModelingError("No partitioned data available")

            partitioned_data = self.data_model.get_partitioned_data()
            X_train = partitioned_data['X_train']
            y_train = partitioned_data['y_train']
            X_test = partitioned_data['X_test']
            y_test = partitioned_data['y_test']

            model_data = self.modeling_service.train_model(
                algorithm=algorithm,
                X_train=X_train,
                y_train=y_train,
                params=params
            )

            self.modeling_model.set_model_data(model_data)

            try:
                onnx_path = self.onnx_service.convert_model_to_onnx(
                    model=model_data.model,
                    sample_input=X_train.iloc[:1] if hasattr(X_train, 'iloc') else X_train[:1],
                    model_name=f"{algorithm}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                self.view.show_success_message(
                    f"Model saved to ONNX format!\n\nFile: {os.path.basename(onnx_path)}\nLocation: {os.path.dirname(onnx_path)}",
                    "Model Saved"
                )
            except Exception as onnx_error:
                self.logger.warning(f"ONNX conversion failed: {onnx_error}")

            try:
                model_type = getattr(model_data, 'task_type', None)
                
                if hasattr(X_test, 'values'):
                    X_test_array = X_test.values
                else:
                    X_test_array = X_test

                if model_type is None:
                    enhanced_processor = EnhancedLabelProcessor()
                    model_type = enhanced_processor.detect_task_type(y_train)

                evaluation_result = None
                try:
                    evaluation_result = self.evaluation_service.evaluate_model(
                        model=model_data.model,
                        X_test=X_test_array,
                        y_test=y_test,
                        task_type=model_type,
                        algorithm=algorithm
                    )
                except Exception as eval_error:
                    error_message = str(eval_error)
                    
                    try:
                        enhanced_processor = EnhancedLabelProcessor()
                        y_train_processed, label_metadata = enhanced_processor.process_labels(y_train, force_task_type=model_type)
                        y_test_processed, _ = enhanced_processor.process_labels(y_test, force_task_type=model_type)
                        
                        evaluation_result = self.evaluation_service.evaluate_model(
                            model=model_data.model,
                            X_test=X_test_array,
                            y_test=y_test_processed,
                            task_type=model_type,
                            algorithm=algorithm
                        )
                    except Exception as secondary_error:
                        self.logger.error(f"Secondary evaluation also failed: {secondary_error}")
                        evaluation_result = None

                if evaluation_result:
                    modeling_result = {
                        'model': model_data.model,
                        'algorithm': algorithm,
                        'task_type': model_type,
                        'evaluation': evaluation_result,
                        'feature_names': getattr(model_data, 'feature_names', None),
                        'training_params': params
                    }

                    predictions = evaluation_result.get('predictions', [])
                    if predictions is not None and len(predictions) > 0:
                        original_label_mapping = getattr(self.data_model, 'label_mapping', None)
                        
                        if original_label_mapping and 'index_to_label' in original_label_mapping:
                            index_to_label = original_label_mapping['index_to_label']
                            mapped_predictions = []
                            for pred in predictions:
                                try:
                                    if isinstance(pred, (int, np.integer)):
                                        mapped_label = index_to_label.get(pred, f"Class_{pred}")
                                    elif isinstance(pred, str) and pred.isdigit():
                                        pred_int = int(pred)
                                        mapped_label = index_to_label.get(pred_int, pred)
                                    else:
                                        mapped_label = str(pred)
                                    mapped_predictions.append(mapped_label)
                                except Exception as map_error:
                                    mapped_predictions.append(str(pred))
                            
                            modeling_result['predictions'] = mapped_predictions
                        else:
                            modeling_result['predictions'] = [str(pred) for pred in predictions]

                    return modeling_result
                else:
                    return {
                        'model': model_data.model,
                        'algorithm': algorithm,
                        'task_type': model_type,
                        'evaluation': None,
                        'feature_names': getattr(model_data, 'feature_names', None),
                        'training_params': params,
                        'predictions': []
                    }

            except Exception as eval_error:
                self.logger.error(f"Model evaluation failed: {eval_error}")
                return {
                    'model': model_data.model,
                    'algorithm': algorithm,
                    'task_type': 'unknown',
                    'evaluation': None,
                    'feature_names': getattr(model_data, 'feature_names', None),
                    'training_params': params,
                    'predictions': []
                }

        except Exception as e:
            raise ModelingError(f"Model training failed: {str(e)}")

    def get_model_data(self):
        return self.modeling_model.get_model_data()

    def has_trained_model(self):
        return self.modeling_model.has_model_data()

    def auto_save_onnx_model(self, trained_model, X_sample, method, evaluation_results):
        try:

            if hasattr(trained_model, 'model'):
                model = trained_model.model
            elif hasattr(trained_model, 'trained_model'):
                model = trained_model.trained_model
            else:
                model = trained_model
            

            if model is None:
                self.logger.warning("No model available for ONNX export")
                return
            

            self.logger.info(f"🚀 Auto-saving model to ONNX format...")
            
            onnx_path = self.onnx_service.auto_save_model(
                model=model,
                X_sample=X_sample,
                model_name=type(model).__name__,
                method=method,
                evaluation_results=evaluation_results
            )
            
            if onnx_path:

                self.view.display_message(
                    f"✅ Model automatically saved to ONNX format!\n\nFile: {os.path.basename(onnx_path)}\nLocation: {os.path.dirname(onnx_path)}", 
                    "Information"
                )
                self.logger.info(f"🎉 ONNX model saved successfully: {onnx_path}")
            else:
                self.logger.info("ℹ️ Model could not be exported to ONNX format (unsupported model type)")
                
        except Exception as e:
            self.logger.error(f"Error in auto_save_onnx_model: {e}")


    def evaluate_model(self, model_id: str, X_test=None, y_test=None):
        print("📊 Starting model evaluation process...")
        
        try:
            model_data = self.modeling_service.get_model_data(model_id)
            if not model_data:
                return {"error": f"Model not found: {model_id}"}
                
            if X_test is None or y_test is None:
                if not hasattr(model_data, 'test_data') or model_data.test_data is None:
                    return {"error": "No test data available"}
                    
                X_test = model_data.test_data.get('X_test')
                y_test = model_data.test_data.get('y_test')
                
                if X_test is None or y_test is None:
                    return {"error": "Test data incomplete"}
            
            print(f"🔍 Model evaluation - Model type: {type(model_data.model).__name__}")
            print(f"   Test data shape: X={X_test.shape}, y={len(y_test) if hasattr(y_test, '__len__') else 'scalar'}")
            print(f"   Test labels sample: {list(y_test[:5]) if hasattr(y_test, '__getitem__') else [y_test]}")
            
            if isinstance(X_test, pd.DataFrame):
                print("✅ Test features are DataFrame")
            elif not isinstance(X_test, np.ndarray):
                X_test = np.array(X_test)
                print("🔄 Converted test features to numpy array")
            

            if hasattr(model_data, 'task_type'):
                model_type = model_data.task_type
                print(f"✅ Task type from model_data: {model_type}")
            elif hasattr(model_data.model, 'task_type'):
                model_type = model_data.model.task_type
                print(f"✅ Task type from model: {model_type}")
            else:

                model_type = self.label_processor.detect_task_type(y_test)
                print(f"🤖 Enhanced auto-detected task type: {model_type}")
            
            try:
                metrics = self.evaluation_service.evaluate_model(
                    model_data.model, X_test, y_test, task_type=model_type)
                
                if isinstance(metrics, dict) and metrics.get('error'):
                    print(f"❌ Evaluation service returned error: {metrics.get('error')}")
                    raise ValueError(metrics.get('error'))
                    
                model_data.evaluation_results = metrics
                self.modeling_service.update_model_data(model_id, model_data)
                return metrics
                
            except Exception as e:
                error_message = str(e)
                print(f"⚠️ Primary evaluation failed: {error_message}")
                
                if "mix of continuous" in error_message or "mix of" in error_message:
                    print("🔄 Attempting to handle mixed label types...")
                    
                    try:

                        y_test_processed, label_metadata = self.label_processor.process_labels_smart(y_test, model_type)
                        print(f"✅ Smart label processing completed: {label_metadata}")
                        
                        metrics = self.evaluation_service.evaluate_model(
                            model_data.model, X_test, y_test_processed, task_type=model_type)
                        print("✅ Evaluation with processed labels successful")
                        
                        if not isinstance(metrics, dict):
                            metrics = {"result": str(metrics), "note": "Used smart label processing for mixed types"}
                        else:
                            metrics["note"] = "Used smart label processing for mixed types"
                            metrics["label_metadata"] = label_metadata
                            
                        model_data.evaluation_results = metrics
                        self.modeling_service.update_model_data(model_id, model_data)
                        return metrics
                    
                    except Exception as backup_error:
                        print(f"❌ Smart label processing also failed: {backup_error}")
                        try:
                            from app.services.evaluation_service import (
                                safe_accuracy, safe_precision, safe_recall, safe_f1
                            )
                            
                            print("🛡️ Attempting safe evaluation functions...")
                            y_pred = model_data.model.predict(X_test)
                            
                            metrics = {
                                "accuracy": float(safe_accuracy(y_test, y_pred)),
                                "precision": float(safe_precision(y_test, y_pred, average='weighted')),
                                "recall": float(safe_recall(y_test, y_pred, average='weighted')),
                                "f1": float(safe_f1(y_test, y_pred, average='weighted')),
                                "task_type": model_type,
                                "note": "Used safe evaluation functions for mixed labels"
                            }
                            
                            model_data.evaluation_results = metrics
                            self.modeling_service.update_model_data(model_id, model_data)
                            return metrics
                            
                        except Exception as final_error:
                            print(f"❌ Safe evaluation also failed: {final_error}")
                            return {"error": f"Evaluation failed: {str(final_error)}", "original_error": error_message}
                
                return {"error": f"Evaluation failed: {error_message}"}
                
        except Exception as e:
            print(f"💥 Evaluation process error: {str(e)}")
            return {"error": f"Evaluation process error: {str(e)}"}

    def apply_modeling(self, method=None, params=None):
        """Apply modeling with the specified method and parameters"""
        try:

            if self.data_model is None:
                raise ValueError("No data available for modeling")
            
            # 🔧 FIX: Use get_modeling_data() to get preprocessed data if available
            modeling_data = self.data_model.get_modeling_data(prefer_selected=True)
            X_train = modeling_data['X_train']
            X_test = modeling_data['X_test']
            y_train = modeling_data['y_train']
            y_test = modeling_data['y_test']
            data_source = modeling_data['source']
            
            print(f"🔧 Modeling data source: {data_source}")
            print(f"   X_train shape: {X_train.shape if X_train is not None else 'None'}")
            print(f"   X_test shape: {X_test.shape if X_test is not None else 'None'}")
            
            if X_train is None or y_train is None:
                raise ValueError("Training data is not available")
            

            original_label_mapping = getattr(self.data_model, 'label_mapping', None)
            print(f"🔧 Original label mapping available: {original_label_mapping is not None}")
            

            modeling_result = self.modeling_service.apply_modeling(
                X_train, y_train, X_test, y_test, method, params
            )
            

            if modeling_result and 'predictions' in modeling_result:
                predictions = modeling_result['predictions']
                print(f"🔧 Raw predictions type: {type(predictions)}, sample: {predictions[:5] if len(predictions) > 5 else predictions}")
                

                if original_label_mapping:
                    index_to_label = original_label_mapping.get('index_to_label', {})
                    if index_to_label:
                        mapped_predictions = []
                        for pred in predictions:
                            try:
                                if isinstance(pred, str):
                                    try:
                                        pred_int = int(float(pred))
                                        mapped_label = index_to_label.get(pred_int, pred)
                                    except (ValueError, TypeError):
                                        mapped_label = pred
                                else:
                                    pred_int = int(round(float(pred)))
                                    mapped_label = index_to_label.get(pred_int, f"Unknown_Class_{pred_int}")
                                mapped_predictions.append(str(mapped_label))
                            except Exception as map_error:
                                print(f"⚠️ Mapping error for prediction {pred}: {map_error}")
                                mapped_predictions.append(str(pred))
                        
                        modeling_result['predictions'] = np.array(mapped_predictions, dtype='<U50')
                        print(f"🔧 Mapped predictions: {modeling_result['predictions'][:5]}")
                    else:

                        modeling_result['predictions'] = np.array([str(pred) for pred in predictions], dtype='<U50')
                        print(f"🔧 No index mapping available, converted to strings")
                else:

                    modeling_result['predictions'] = np.array([str(pred) for pred in predictions], dtype='<U50')
                    print(f"🔧 No mapping available, converted to strings")
            

            self.data_model.modeling_result = modeling_result
            
            return modeling_result
            
        except Exception as e:
            error_msg = f"Modeling failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.logger.error(error_msg)
            raise Exception(error_msg)
