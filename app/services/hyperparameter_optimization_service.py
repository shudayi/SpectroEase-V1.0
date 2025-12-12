# app/services/hyperparameter_optimization_service.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
import numpy as np
import random
from app.utils.logger import setup_logger
from app.utils.exceptions import HyperparameterOptimizationError
# **CRITICAL FIX: Import enhanced label processor for consistent handling**
from app.utils.label_processor import EnhancedLabelProcessor

class HyperparameterOptimizationService:
    def __init__(self):
        self.logger = setup_logger()
        # **CRITICAL FIX: Initialize enhanced label processor for consistent handling**
        self.label_processor = EnhancedLabelProcessor()
        print("🔧 HyperparameterOptimizationService initialized with EnhancedLabelProcessor")

    def no_optimization(self, estimator, X, y):
        """No hyperparameter optimization, directly return original model."""
        self.logger.info("No hyperparameter optimization applied.")
        
        # **CRITICAL FIX: Use enhanced label processor for consistent handling**
        # Check if this is a classification estimator
        is_classifier = hasattr(estimator, 'predict_proba') or 'Classifier' in type(estimator).__name__
        
        if is_classifier:
            # Use enhanced label processor for smart label handling
            task_type = self.label_processor.detect_task_type(y)
            if task_type == 'classification':
                y_processed, label_metadata = self.label_processor.process_labels_smart(y, 'classification')
                print(f"✅ Enhanced label processing: {label_metadata}")
                estimator.fit(X, y_processed)
            else:
                # Fallback for regression-like data
                estimator.fit(X, y)
        else:
            estimator.fit(X, y)
        return estimator

    def grid_search(self, estimator, param_grid, X, y, cv=5):
        """Execute grid search optimization."""
        try:
            # **CRITICAL FIX: Use enhanced label processor for consistent handling**
            # Check if this is a classification estimator
            is_classifier = hasattr(estimator, 'predict_proba') or 'Classifier' in type(estimator).__name__
            
            if is_classifier:
                # Use enhanced label processor for smart label handling
                task_type = self.label_processor.detect_task_type(y)
                if task_type == 'classification':
                    y_processed, label_metadata = self.label_processor.process_labels_smart(y, 'classification')
                    print(f"✅ Enhanced label processing for grid search: {label_metadata}")
                    
                    grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
                    grid_search.fit(X, y_processed)
                else:
                    # Fallback for regression-like data
                    grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
                    grid_search.fit(X, y)
            else:
                grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
                grid_search.fit(X, y)
            
            self.logger.info("Grid Search optimization completed.")
            return grid_search.best_estimator_
        except Exception as e:
            self.logger.error(f"Grid Search failed: {e}")
            raise HyperparameterOptimizationError(f"Grid Search failed: {e}")

    def random_search(self, estimator, param_distributions, X, y, cv=5, n_iter=10):
        """Execute random search optimization."""
        try:
            # **CRITICAL FIX: Use enhanced label processor for consistent handling**
            # Check if this is a classification estimator
            is_classifier = hasattr(estimator, 'predict_proba') or 'Classifier' in type(estimator).__name__
            
            if is_classifier:
                # Use enhanced label processor for smart label handling
                task_type = self.label_processor.detect_task_type(y)
                if task_type == 'classification':
                    y_processed, label_metadata = self.label_processor.process_labels_smart(y, 'classification')
                    print(f"✅ Enhanced label processing for random search: {label_metadata}")
                    
                    random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=n_iter, cv=cv, scoring='accuracy', random_state=42)
                    random_search.fit(X, y_processed)
                else:
                    # Fallback for regression-like data
                    random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=n_iter, cv=cv, scoring='accuracy', random_state=42)
                    random_search.fit(X, y)
            else:
                random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=n_iter, cv=cv, scoring='accuracy', random_state=42)
                random_search.fit(X, y)
            
            self.logger.info("Random Search optimization completed.")
            return random_search.best_estimator_
        except Exception as e:
            self.logger.error(f"Random Search failed: {e}")
            raise HyperparameterOptimizationError(f"Random Search failed: {e}")

    def genetic_algorithm(self, estimator_class, param_ranges, X, y, cv=5, n_population=20, n_generations=10):
        """Use genetic algorithm for hyperparameter optimization."""
        try:
  
            def eval_estimator(individual):
  
                params = {}
                for key, val in zip(param_ranges.keys(), individual):
                    if isinstance(param_ranges[key], list):
                        params[key] = param_ranges[key][val % len(param_ranges[key])]
                    else:
                        params[key] = val

                estimator = estimator_class(**params)
                scores = []
                # **CRITICAL FIX: Handle mixed label types in genetic algorithm**
                # Standardize labels to prevent mixed type errors
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import LabelEncoder
                
                if isinstance(y, pd.Series):
                    y_safe = y.astype(str)
                else:
                    y_safe = np.array([str(label) for label in y])
                
                # Encode to integers for sklearn
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_safe)
                if isinstance(y, pd.Series):
                    y_encoded = pd.Series(y_encoded, index=y.index)
                
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                for train_idx, test_idx in skf.split(X, y_encoded):
                    estimator.fit(X.iloc[train_idx], y_encoded.iloc[train_idx] if isinstance(y_encoded, pd.Series) else y_encoded[train_idx])
                    preds = estimator.predict(X.iloc[test_idx])
                    actual = y_encoded.iloc[test_idx] if isinstance(y_encoded, pd.Series) else y_encoded[test_idx]
                    scores.append(accuracy_score(actual, preds))
                return (np.mean(scores),)

  
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

            toolbox = base.Toolbox()

  
            for key, val in param_ranges.items():
                if isinstance(val, list):
                    toolbox.register(f"attr_{key}", random.choice, val)
                elif isinstance(val, tuple):
                    toolbox.register(f"attr_{key}", random.uniform, val[0], val[1])

  
            attr_names = [f"attr_{key}" for key in param_ranges.keys()]
            toolbox.register("individual", tools.initCycle, creator.Individual,
                             (toolbox.__getattribute__(name) for name in attr_names), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", eval_estimator)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)

            population = toolbox.population(n=n_population)
            algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=False)

  
            best_individual = tools.selBest(population, k=1)[0]
            best_params = {}
            for key, val in zip(param_ranges.keys(), best_individual):
                if isinstance(param_ranges[key], list):
                    best_params[key] = param_ranges[key][val % len(param_ranges[key])]
                else:
                    best_params[key] = val

            best_estimator = estimator_class(**best_params)
            
            # **CRITICAL FIX: Handle mixed label types for final fit**
            # Standardize labels to prevent mixed type errors
            if isinstance(y, pd.Series):
                y_safe = y.astype(str)
            else:
                y_safe = np.array([str(label) for label in y])
            
            # Encode to integers for sklearn
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_safe)
            best_estimator.fit(X, y_encoded)
            
            self.logger.info("Genetic Algorithm optimization completed.")
            return best_estimator
        except Exception as e:
            self.logger.error(f"Genetic Algorithm optimization failed: {e}")
            raise HyperparameterOptimizationError(f"Genetic Algorithm optimization failed: {e}")
