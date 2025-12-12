from typing import Dict, Any
import json
import re
import asyncio
from config.llm_config import LLMConfig
import aiohttp  # V1.3.1: Use async HTTP client instead of requests


class LLMService:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = None):
        self.api_key = api_key
        # 🔧 FIX: Use configured base_url instead of hardcoded value
        # Normalize URL: remove trailing slash to avoid double slashes
        raw_url = base_url or LLMConfig.API_BASE_URL or "https://api.deepseek.com/v1"
        self.base_url = raw_url.rstrip('/') if raw_url else "https://api.deepseek.com/v1"
        self.model_name = model_name or LLMConfig.MODEL_NAME or "deepseek-coder"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        print(f"🤖 LLM Service initialized:")
        print(f"   Base URL: {self.base_url}")
        print(f"   Model: {self.model_name}")
    
    def clean_code(self, code: str) -> str:
        """
        Clean LLM returned code, remove markdown format and other invalid content
        V1.3.1: Add code cleaning functionality
        """
        if not code:
            return code
        
        # Remove markdown code block markers
        # Match ```python ... ``` or ```...```
        code = re.sub(r'^```(?:python)?\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove possible extra explanatory text (usually at the beginning or end)
        lines = code.split('\n')
        cleaned_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip common explanatory text
            if stripped.startswith(('Here is', "Here's", 'This is', 'The following', '以下是', '这是')):
                continue
            
            # Detect import or class start, consider code started
            if stripped.startswith(('import ', 'from ', 'class ', 'def ', '@')):
                in_code = True
            
            if in_code or stripped.startswith('#'):
                cleaned_lines.append(line)
        
        cleaned_code = '\n'.join(cleaned_lines).strip()
        
        return cleaned_code

    def _build_algorithm_context(self, algorithm_type: str) -> str:
        """
        Build algorithm-specific context for user prompt.
        This provides critical information about data format and expected behavior
        for each algorithm type.
        """
        context_map = {
            'Preprocessing': """This is a PREPROCESSING algorithm for spectral/chemometric data.

CRITICAL DATA FORMAT REQUIREMENTS (ACTUAL SYSTEM FORMAT):
- The apply() method will receive a DataFrame where:
  * EACH ROW represents a SAMPLE/SPECTRUM
  * EACH COLUMN represents a WAVELENGTH/FEATURE point
- You MUST preserve this layout: rows = samples, columns = features/wavelengths
- The output DataFrame MUST have the same shape and index/columns as input
- This matches the system's actual data format: X_train/X_test are (samples × features) DataFrames

CRITICAL ALGORITHM-SPECIFIC NOTES:
- For MSC (Multiplicative Scatter Correction):
  * IMPORTANT: MATLAB polyfit(x,y,1) returns [slope, intercept] (p(1)=slope, p(2)=intercept)
  * numpy.polynomial.polynomial.polyfit(x,y,1) returns [intercept, slope] (p[0]=intercept, p[1]=slope)
  * When converting from MATLAB: if original uses p(2) and p(1), convert to p[0] and p[1] in numpy
  * MATLAB MSC: (y - p(2)) / p(1) → Python: (y - p[0]) / p[1] where p[0]=intercept, p[1]=slope
  * Always verify coefficient order matches the library being used
- For Savitzky-Golay filtering:
  * Preserve the EXACT deriv parameter from original code (deriv=0, 1, or 2)
  * MATLAB sgolayfilt(..., deriv=1) → Python: savgol_filter(..., deriv=1)
  * deriv=0: smoothing only, deriv=1: first derivative (common in MATLAB), deriv=2: second derivative
  * Do NOT silently change derivative order

If the original code uses a different layout (e.g., columns = samples in MATLAB/R),
you MUST transpose internally for calculations, but MUST return data in the system format:
rows = samples, columns = features/wavelengths.""",

            'Feature Selection': """This is a FEATURE SELECTION algorithm.

CRITICAL DATA FORMAT REQUIREMENTS:
- The select_features() method will receive:
  * X: DataFrame where ROWS = samples, COLUMNS = wavelengths/features
  * y: Series with target values (one per sample)
  
CRITICAL RETURN FORMAT (HARD REQUIREMENT):
- You MUST return a List[str] containing COLUMN NAMES from X.columns
- NEVER return:
  * Row indices (e.g., X.index[...])
  * Sample names
  * Numeric indices (e.g., [0, 1, 2])
  * Any other format
- ALWAYS use: return [X.columns[i] for i in selected_indices]
- This is a SYSTEM REQUIREMENT - the downstream code expects column names

COMMON PITFALLS TO AVOID:
- If original code uses columns = samples layout:
  * You may transpose X internally for calculations: X_array = X.values.T
  * But you MUST map final selected indices back to X.columns names
  * Example: selected_features = [X.columns[i] for i in selected_indices]
- If using PCA loadings or similar:
  * Ensure you're selecting features (columns), not samples (rows)
  * Double-check that selected_indices refer to column positions, not row positions""",

            'Modeling': """This is a MODELING algorithm (regression or classification).

CRITICAL DATA FORMAT REQUIREMENTS:
- The train() method will receive:
  * X: DataFrame where ROWS = samples, COLUMNS = selected wavelengths/features
  * y: Series with target values (one per sample, same length as X)
- The predict() method will receive:
  * model: The object returned by train()
  * X: DataFrame with same column structure as training X

CRITICAL HYPERPARAMETER SELECTION RULE (HARD REQUIREMENT):
- When selecting best hyperparameters from cross-validation scores (e.g., best_ncomp, best_k):
  * You MUST use: best_k = int(np.argmin(cv_scores)) + 1
  * This is a SYSTEM REQUIREMENT - no exceptions unless original code explicitly does otherwise
- FORBIDDEN patterns (DO NOT USE):
  * sel[np.argmin(cv_scores[1:]) + 1]  ← WRONG: complex indexing
  * range(...)[np.argmin(cv_scores[1:]) + 1]  ← WRONG: slicing scores array
  * Any slicing/shifting of cv_scores array before argmin
- The +1 is because argmin returns 0-based index, but component numbers are 1-based
- This applies to ALL algorithms using cross-validation (PLS, PCR, etc.)

If the original code uses a different layout, transpose internally but ensure
the final model operates on rows = samples, columns = features.""",

            'Data Partitioning': """This is a DATA PARTITIONING algorithm.

CRITICAL DATA FORMAT REQUIREMENTS:
- The partition() method will receive:
  * data: DataFrame where ROWS = samples, COLUMNS = features, LAST COLUMN = target/label
- You MUST extract: X = data.iloc[:, :-1], y = data.iloc[:, -1]
- You MUST return: X_train, X_test, y_train, y_test (all as DataFrame/Series)

CRITICAL LOGIC PRESERVATION:
- If the original code splits data along a different axis (e.g., columns = samples):
  * Original: spectra is (wavelengths × samples), ron is separate vector
  * New system: data rows = samples, last column = target
  * You MUST rewrite the logic to operate on rows as samples
  * Preserve the same split behavior: same random_state, test_size, and stratification
  * Ensure the split indices match the original behavior (same samples in train/test)
- The data layout assumption has changed from original code, but split logic must remain equivalent"""
        }
        
        return context_map.get(algorithm_type, "")

    async def chat(self, message: str, code: str = None, algorithm_type: str = None) -> str:
        """
        Chat with the LLM
        V1.4.3: Add algorithm_type parameter to build algorithm-specific context
        
        Args:
            message: User message
            code: Original algorithm code to convert
            algorithm_type: One of 'Preprocessing', 'Feature Selection', 'Modeling', 'Data Partitioning'
        """
        # Build algorithm-specific context
        algorithm_context = ""
        if algorithm_type:
            algorithm_context = self._build_algorithm_context(algorithm_type)
        
        # Build prompt
        if code:
            prompt = f"""User message:
{message}

{algorithm_context}

Original algorithm code (for conversion):
{code}

Task:
Convert the above algorithm into a single Python class that implements ONE of the system interfaces
(PreprocessingAlgorithm, FeatureSelectionAlgorithm, ModelingAlgorithm, or DataPartitioningAlgorithm),
and preserve the original algorithm logic.

Requirements:
1. Code follows Python syntax standards
2. Code adheres to interface definition requirements
3. Maintains the core logic and math of the original algorithm
4. Provides complete code implementation without omitting any parts
5. Ensures the class can be directly integrated into the system
"""
        else:
            prompt = f"""User message:
{message}

{algorithm_context}

Please provide a professional response. If code is involved, ensure it:
1. Follows Python syntax standards
2. Adheres to the interface definition requirements
3. Provides a complete code implementation without omitting any parts
4. Can be directly integrated into the system
"""
        
        try:
            # V1.3.1: Use aiohttp for true async HTTP calls
            # V1.4.3: Add timeout to prevent hanging
            timeout = aiohttp.ClientTimeout(total=120)  # 120 seconds timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are a spectroscopy and chemometrics algorithm conversion expert.

Your job:
Convert given algorithm code (Python / MATLAB / R / C++ / etc.) into EXACTLY ONE Python class that implements a strict interface used in our system.

==================================================
0. FINAL OUTPUT FORMAT (CRITICAL)
==================================================

- Output MUST be ONLY valid Python code for a single module.
- DO NOT include any markdown, backticks, or explanatory text.
- Do NOT wrap the code in code fences.
- Comments inside the Python code are allowed.
- The result must be a complete, importable Python file.

==================================================
1. INTERFACE TYPE SELECTION (MUST CHOOSE ONE)
==================================================

Decide which interface to use based on WHAT THE ORIGINAL ALGORITHM DOES.
Choose exactly ONE interface and implement exactly ONE class.

A. PreprocessingAlgorithm
   Use this when:
   - The algorithm processes input data X (e.g. spectra, feature matrix).
   - It applies transformations: baseline correction, smoothing, normalization,
     scaling, scatter correction, derivatives, etc.
   - It returns transformed X only (no model object, no training).
   Required skeleton:
     class YourPreprocessor(PreprocessingAlgorithm):
         def get_name(self) -> str: ...
         def get_params_info(self) -> Dict[str, Any]: ...
         def apply(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame: ...

B. FeatureSelectionAlgorithm
   Use this when:
   - The algorithm selects a subset of variables/features/wavelengths.
   - It may use X alone or both X and y (supervised selection).
   - The result is a list of selected feature names (column names of X).
   Required skeleton:
     class YourSelector(FeatureSelectionAlgorithm):
         def get_name(self) -> str: ...
         def get_params_info(self) -> Dict[str, Any]: ...
         def select_features(self, X: pd.DataFrame, y: pd.Series, params: Dict) -> List[str]: ...
   
   ⚠️ CRITICAL RULE - FEATURE SELECTION RETURN FORMAT:
   - select_features() MUST return List[str] containing COLUMN NAMES from X.columns
   - NEVER return row indices, sample names, or numeric indices
   - NEVER return X.index[...] or sample names
   - ALWAYS use: return [X.columns[i] for i in selected_indices]
   - This is a HARD REQUIREMENT that applies to ALL feature selection algorithms

C. ModelingAlgorithm
   Use this when:
   - The algorithm trains a predictive model (regression or classification).
   - There is a training phase and a prediction phase.
   - Examples: PLS, PCR, regression, SVM, neural networks, etc.
   Required skeleton:
     class YourModel(ModelingAlgorithm):
         def get_name(self) -> str: ...
         def get_params_info(self) -> Dict[str, Any]: ...
         def train(self, X: pd.DataFrame, y: pd.Series, params: Dict) -> Any: ...
         def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray: ...

D. DataPartitioningAlgorithm
   Use this when:
   - The algorithm splits data into train/test or cross-validation subsets.
   - There is no modeling or preprocessing, only partitioning.
   Required skeleton:
     class YourPartitioner(DataPartitioningAlgorithm):
         def get_name(self) -> str: ...
         def get_params_info(self) -> Dict[str, Any]: ...
         def partition(self, data: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: ...

==================================================
2. PARAMETER SPECIFICATION (get_params_info)
==================================================

The method get_params_info() MUST return a dict like:

{
  "param_name": {
    "type": "int" | "float" | "str" | "bool",
    "default": <default_value>,
    "description": "human readable description"
  },
  ...
}

Rules:
- Include all important tunable parameters present in the original algorithm.
- Default values must match the original code whenever possible.
- Descriptions must clearly explain the meaning and role of each parameter.

==================================================
3. DATA FLOW & FORMAT REQUIREMENTS (CRITICAL)
==================================================

⚠️ CRITICAL: Your converted code MUST integrate seamlessly into the system's data pipeline.

A) Data Format Standards:
   - ALL data inputs/outputs use pandas DataFrame/Series
   - DataFrame: rows = samples (样本), columns = features (特征/波长)
   - Series: one value per sample (length = DataFrame.shape[0])
   - Index: MUST be preserved - never lose or reorder index
   - Columns: feature names MUST be preserved

B) Data Flow Pipeline:
   1. Data Loading → DataFrame (rows=samples, columns=features+label)
   2. Data Partitioning → X_train, X_test (DataFrame), y_train, y_test (Series)
   3. Preprocessing → DataFrame (same shape, same index/columns)
   4. Feature Selection → List[str] (column names from X.columns)
   5. Modeling → train() returns model, predict() returns np.ndarray

C) Interface-Specific Data Flow Requirements:

   PreprocessingAlgorithm.apply():
   - Input: data (pd.DataFrame, rows=samples, columns=features)
   - Output: pd.DataFrame (SAME shape, SAME index, SAME columns)
   - CRITICAL: If you transpose internally for calculations, you MUST transpose back
   - CRITICAL: Always preserve index and columns:
     result = pd.DataFrame(processed_array, index=data.index, columns=data.columns)
   - Example pattern:
     original_index = data.index
     original_columns = data.columns
     # ... do processing (may transpose internally) ...
     result_array = ...  # Final result as numpy array
     return pd.DataFrame(result_array, index=original_index, columns=original_columns)

   FeatureSelectionAlgorithm.select_features():
   - Input: X (pd.DataFrame, rows=samples, columns=features), y (pd.Series)
   - Output: List[str] (selected feature COLUMN NAMES)
   - CRITICAL: MUST return column names, NOT indices or sample names
   - CRITICAL: Use: return [X.columns[i] for i in selected_indices]
   - FORBIDDEN: return [i for i in ...]  ← numeric indices
   - FORBIDDEN: return X.index[...]  ← sample names
   - FORBIDDEN: return list(range(...))  ← numeric indices

   ModelingAlgorithm.train():
   - Input: X (pd.DataFrame, rows=samples, columns=features), y (pd.Series)
   - Output: Any (typically dict containing 'model' key and other metadata)
   - CRITICAL: Model object should be serializable (for saving/loading)
   - Common pattern: return {'model': trained_model, 'best_params': ..., 'cv_scores': ...}

   ModelingAlgorithm.predict():
   - Input: model (from train()), X (pd.DataFrame, rows=samples, columns=features)
   - Output: np.ndarray (1D array, length = X.shape[0])
   - CRITICAL: Output MUST be numpy array, NOT DataFrame or Series
   - CRITICAL: Length MUST match number of samples: len(output) == X.shape[0]
   - Pattern: return predictions.flatten() if 2D, or predictions if 1D

   DataPartitioningAlgorithm.partition():
   - Input: data (pd.DataFrame, rows=samples, columns=features+label)
   - Output: Tuple[X_train, X_test, y_train, y_test]
     * X_train, X_test: pd.DataFrame (rows=samples, columns=features)
     * y_train, y_test: pd.Series (one value per sample)
   - CRITICAL: Must return exactly 4 objects in this order
   - CRITICAL: Preserve index from original data
   - Pattern: X = data.iloc[:, :-1], y = data.iloc[:, -1]

D) Index and Column Preservation Rules:
   - NEVER use reset_index(drop=True) unless absolutely necessary
   - ALWAYS preserve original index when creating new DataFrames
   - ALWAYS preserve original columns when creating new DataFrames
   - If you must reindex, document why and ensure it doesn't break data flow

E) Data Type Consistency:
   - Input types: pandas DataFrame/Series (never numpy arrays as input)
   - Output types: 
     * Preprocessing: DataFrame (not numpy array)
     * Feature Selection: List[str] (not numpy array, not indices)
     * Modeling train: Any (dict recommended)
     * Modeling predict: np.ndarray (not DataFrame, not Series)
   - Internal calculations: Can use numpy arrays, but convert back to DataFrame for output

F) Shape Verification:
   - Preprocessing: Input and output shapes MUST match
   - Feature Selection: Output list length = number of selected features
   - Modeling predict: Output array length = input X.shape[0]
   - Add shape verification comments in code:
     # Verify: input shape (n_samples, n_features) → output shape (n_samples, n_features)

==================================================
4. GLOBAL IMPLEMENTATION RULES
==================================================

1) Class inheritance:
   - The class MUST inherit from the correct base class:
       class Xxx(PreprocessingAlgorithm):
       class Xxx(FeatureSelectionAlgorithm):
       class Xxx(ModelingAlgorithm):
       class Xxx(DataPartitioningAlgorithm):

2) Imports:
   - Always import at least:
       import pandas as pd
       import numpy as np
       from typing import Dict, Any, List, Tuple
   - Import any additional libraries that the algorithm actually uses
     (e.g., scipy, sklearn), but avoid unused imports.

3) Required methods:
   - Implement ALL abstract methods for the chosen interface with the
     exact signatures described above.
   - Do not change method names or signatures.

4) __init__:
   - Either no __init__ method, or an __init__ that uses only default
     values and does not require external arguments.
   - Do not rely on external global state.

5) No file I/O:
   - Do NOT read from or write to files (CSV, TXT, etc.).
   - Do NOT print debug information.
   - The class must operate only on the method inputs and outputs.

6) Data format assumptions:
   - Unless the original code clearly uses a different convention,
     assume:
       * rows = samples
       * columns = features for X
       * last column = target/label when using DataPartitioningAlgorithm
   - If the original code uses a different layout (e.g., columns = samples),
     you may transpose data internally but MUST preserve the same mathematical
     behavior AND restore the original format in the output.

==================================================
5. CODE ANALYSIS & VERIFICATION (MANDATORY FIRST STEP)
==================================================

BEFORE writing the converted code, you MUST analyze the original algorithm:

STEP 1: Identify all mathematical operations:
- List all division operations (a / b) - these need numerical stability checks
- List all matrix multiplications (A @ B, A.T @ B, etc.) - verify dimensions
- List all matrix decompositions (eigen, SVD, PCA) - verify what is being computed
- List all polynomial/regression fits - verify coefficient order
- List all statistical operations (mean, std, covariance) - verify axis/dimension

STEP 2: Identify potential numerical issues:
- Division operations: Check if denominator can be zero or near-zero
- Matrix operations: Verify matrix dimensions match mathematical intent
- Transpose operations: Ensure transposes preserve mathematical meaning
- Index operations: Verify 0-based vs 1-based indexing conversions

STEP 3: Identify data layout assumptions:
- What is the original data shape? (rows = ?, columns = ?)
- Does the algorithm transpose data internally?
- What are the expected dimensions of intermediate results?

STEP 4: Map operations to Python equivalents:
- For each mathematical operation, identify the correct Python/NumPy equivalent
- Verify coefficient orders for polynomial fits
- Verify matrix multiplication order for covariance/decomposition
- Verify axis parameters for statistical operations

==================================================
6. PRESERVE ORIGINAL ALGORITHM LOGIC (CRITICAL)
==================================================

You MUST preserve the original algorithm's logic as exactly as possible.

- Keep all mathematical operations and formulas equivalent.
- Keep the order of processing steps the same.
- Preserve all parameter meanings and default values.
- Preserve all implicit assumptions about data shape and indexing.
- Do NOT simplify, optimize, or modify the core math.
- You may refactor code structure (e.g., loops, helper variables) as long as
  the numerical results remain the same.

When converting from MATLAB / R / C++:
- Pay close attention to matrix shapes (rows vs columns) and transposes.
- Resolve differences in indexing (MATLAB/R: 1-based, Python: 0-based).
- When converting operations like matrix multiplication, covariance, eigen
  decomposition, or regressions, make sure the resulting Python expressions
  compute the same mathematical quantity.

==================================================
7. UNIVERSAL NUMERICAL STABILITY CONSTRAINTS (MANDATORY)
==================================================

⚠️ CRITICAL: These rules apply to ALL algorithms, regardless of type.

A) Division Operations (MANDATORY CHECK):
   - For EVERY division operation (a / b) in your code:
     * You MUST check if denominator can be zero or near-zero
     * Add numerical stability check BEFORE division:
       Example pattern:
         denominator = <some_value>
         if abs(denominator) < 1e-8:  # or appropriate threshold
             # Handle edge case: use fallback or skip operation
             result = <safe_fallback_value>
         else:
             result = numerator / denominator
     * This applies to:
       - MSC corrections: (y - intercept) / slope
       - Normalizations: x / norm, x / std
       - Any ratio calculations: a / b
       - Matrix inversions: A / det, etc.
   - Threshold selection:
     * Use 1e-8 for most cases
     * Use 1e-10 for very precise calculations
     * Adjust based on data scale if needed

B) Matrix Operations (MANDATORY VERIFICATION):
   - For EVERY matrix multiplication (A @ B, A.T @ B, etc.):
     * BEFORE writing the operation, verify:
       1. What mathematical quantity are you computing?
       2. What should the output shape be?
       3. Does the operation match the mathematical intent?
     * Common patterns:
       - Covariance matrix (features × features):
         * If X shape is (n_features, n_samples):
           CORRECT: Cov = (X @ X.T) / (n_samples - 1)
           WRONG: Cov = (X.T @ X) / (n_samples - 1)  ← This gives samples×samples!
       - Cross-covariance (features × features):
         * Verify: result shape = (n_features, n_features)
         * NOT (n_samples, n_samples)
     * Always add shape verification comments:
       # Verify: X shape (n_features, n_samples) → Cov shape (n_features, n_features)
       Cov = (X_centered @ X_centered.T) / (n_samples - 1)
       assert Cov.shape == (n_features, n_features), f"Expected ({n_features}, {n_features}), got {Cov.shape}"

C) Polynomial/Regression Fitting (MANDATORY VERIFICATION):
   - For EVERY polynomial fit or linear regression:
     * Identify which function is used:
       - numpy.polyfit → returns [slope, intercept] (high to low order)
       - numpy.polynomial.polynomial.polyfit → returns [intercept, slope] (low to high order)
     * BEFORE using coefficients:
       1. Check the function's documentation for coefficient order
       2. Verify how the original code uses the coefficients
       3. Reconstruct the polynomial correctly
     * Example verification:
       p = polyfit(x, y, 1)  # Check: what order does this return?
       # Test: y_reconstructed = p[0] + p[1]*x  OR  p[1] + p[0]*x?
       # Verify against original code's usage

D) Statistical Operations (MANDATORY AXIS VERIFICATION):
   - For EVERY statistical operation (mean, std, sum, etc.):
     * Verify the axis parameter matches the mathematical intent:
       - axis=0: operation along rows (across samples)
       - axis=1: operation along columns (across features)
     * If original code uses different convention, adapt correctly
     * Add comments explaining axis choice:
       # Mean across samples (axis=0) → shape (n_features,)
       mean_spectrum = np.mean(X, axis=0)

E) Decomposition Operations (MANDATORY COMPONENT VERIFICATION):
   - For EVERY decomposition (PCA, SVD, eigen, etc.):
     * Verify what is being computed:
       - Feature covariance vs sample covariance
       - Which components/vectors are selected
       - How loadings/coefficients are interpreted
     * Add verification comments:
       # PCA on feature covariance: X shape (n_features, n_samples)
       # → Cov shape (n_features, n_features) → eigenvectors shape (n_features, n_components)
       Cov = (X_centered @ X_centered.T) / (n_samples - 1)
       evals, evecs = np.linalg.eigh(Cov)
       # evecs[:, i] is the i-th principal component (feature loadings)

F) Hyperparameter Selection (MANDATORY PATTERN):
   - For EVERY cross-validation based hyperparameter selection:
     * Use the standard pattern: best_k = int(np.argmin(cv_scores)) + 1
     * DO NOT use complex indexing or array slicing
     * The +1 converts 0-based index to 1-based component number

==================================================
8. COMMON PITFALLS AND HOW TO HANDLE THEM
==================================================

These are general examples. Apply them ONLY when the original algorithm
actually uses the corresponding operations.

A) Polynomial fitting equivalents:
   - Different libraries may return polynomial coefficients in different orders.
   - Before using coefficients, verify:
       * which function is used (e.g., numpy.polyfit vs numpy.polynomial.polynomial.polyfit),
       * the documented coefficient order,
       * how the original language defines the polynomial.
   - Always reconstruct the fitted value using the correct coefficient order
     so that the result matches the original implementation.

B) Derivatives and filter parameters:
   - Many filtering / smoothing functions accept a derivative order or similar parameter.
   - If the source algorithm specifies a derivative order or filter mode,
     you MUST keep the same value in the Python version.
   - Do NOT silently change a derivative order (e.g., from first derivative
     to pure smoothing).

C) Decomposition and component selection:
   - For algorithms using eigenvalues, singular values, or components
     (e.g., PCA, SVD-based methods, latent variable models):
       * Always select the correct eigenvector / component associated with
         the intended eigenvalue or singular value (usually the largest).
       * When ranking features by loadings or contributions, use the same
         rule as the original (e.g., absolute value, descending order).

D) Randomness and reproducibility:
   - If the original code sets a random seed or uses a specific random
     split strategy, do the same in Python.
   - Use parameters like random_state in scikit-learn when relevant.

E) Data partitioning:
   - When splitting data (train/test or folds), match:
       * the ratio or sizes (e.g., test_size),
       * any stratification behavior,
       * and any fixed random seed used in the original implementation.
   - The partition() method must return:
       X_train, X_test, y_train, y_test
     as pandas DataFrame/Series objects consistent with the interface.

==================================================
9. CODE STYLE & COMPLETENESS
==================================================

- Use clear, descriptive class names that reflect the algorithm, e.g.:
   BaselineCorrectionPreprocessor
   WaveletDenoisingPreprocessor
   PCALoadingFeatureSelector
   GenericRegressionModel
   SimpleTrainTestSplitter

- The final code must be self-contained and directly usable in the system:
   * No missing methods.
   * No placeholder "pass" statements.
   * All referenced symbols must be imported or defined.

==================================================
10. SHAPE & DATA LAYOUT DOCUMENTATION (REQUIRED)
==================================================

At the top of your class implementation (right after imports, before the class definition),
include a comment block that explicitly documents:

1. Expected input data shape and meaning (ACTUAL SYSTEM FORMAT):
   - For PreprocessingAlgorithm: 
     * Input: data (pd.DataFrame, rows=samples, columns=features)
     * Output: data (pd.DataFrame, SAME shape, SAME index, SAME columns)
     * Example: data.shape = (n_samples, n_wavelengths)
     * CRITICAL: Must preserve index and columns
   - For FeatureSelectionAlgorithm: 
     * Input: X (pd.DataFrame, rows=samples, columns=features), y (pd.Series)
     * Output: List[str] (column names from X.columns)
     * CRITICAL: Must return column names, not indices
   - For ModelingAlgorithm: 
     * Input: X (pd.DataFrame, rows=samples, columns=features), y (pd.Series)
     * Output: train() → Any (model object), predict() → np.ndarray (length=X.shape[0])
     * CRITICAL: predict() output must be 1D numpy array
   - For DataPartitioningAlgorithm: 
     * Input: data (pd.DataFrame, rows=samples, columns=features+target)
     * Output: Tuple[X_train, X_test, y_train, y_test] (4 objects)
     * CRITICAL: Must preserve index, return exactly 4 objects

2. Output format:
   - What does the output represent? (e.g., "selected feature names from X.columns")
   - What is the shape/format of the return value?

3. Relationship to original implementation:
   - If the original code used a different data layout (e.g., columns = samples),
    briefly note how you adapted it while preserving the mathematical behavior.

Example comment block (note: this is Python code, not markdown):
# Data Format Assumptions:
# - Input X: rows = samples, columns = features/wavelengths
# - Input y: Series with one target value per sample (same length as X)
# - Output: List[str] of selected feature names from X.columns
#
# Original Implementation:
# - Original code used columns = samples, rows = features
# - Adapted by transposing X internally, then mapping selected indices back to X.columns

This documentation helps verify that the conversion correctly handles data layout.

REMINDER:
- FINAL ANSWER = ONLY the complete Python code for the class/module.
- NO markdown, NO backticks, NO natural-language explanation outside comments.
- Include the shape documentation as Python comments at the top of the class.
"""
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": LLMConfig.TEMPERATURE,
                        "max_tokens": LLMConfig.MAX_TOKENS
                    }
                ) as response:
                    # V1.3.1: aiohttp uses response.status and await response.json()
                    if response.status == 200:
                        result_data = await response.json()
                        raw_code = result_data["choices"][0]["message"]["content"]
                        # V1.3.1: Clean code, remove markdown format
                        cleaned_code = self.clean_code(raw_code)
                        return cleaned_code
                    else:
                        error_text = await response.text()
                        raise Exception(f"API call failed: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            print(f"❌ Network error calling Deepseek API: {str(e)}")
            return None
        except asyncio.TimeoutError:
            print(f"❌ Request timeout: LLM API call took longer than 120 seconds")
            print(f"   This may happen with complex code conversions. Try simplifying the code or increasing timeout.")
            return None
        except Exception as e:
            print(f"❌ Error calling Deepseek API: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
