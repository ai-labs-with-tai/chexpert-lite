You are a senior Python developer helping refactor a university student project.

This project is a Digital Image Processing / Medical Image Classification demo using Streamlit, PyTorch, ResNet50, DenseNet121, CheXpert models, prediction threshold, and Grad-CAM visualization.

Your task is to refactor the code to make it easier to understand, easier to explain, and suitable for a student project.

This is NOT an enterprise project.

The most important goal is:

> A lecturer or student should be able to open the project and understand what each part does.

Do not over-engineer.

Do not create too many files.

Do not introduce complex architecture.

Preserve the current behavior exactly.

---

# Main Refactor Goal

Refactor the current code so that responsibilities are separated clearly:

- Streamlit UI
- model loading
- image preprocessing
- prediction logic
- Grad-CAM visualization
- disease labels / Vietnamese names
- configuration paths

The final code should look like a careful student wrote it, not like a large software company project.

---

# Suggested Simple Structure

Use a simple structure like this:

```text
app.py
src/
    config.py
    disease_info.py
    model_loader.py
    image_processing.py
    prediction.py
    gradcam.py
    ui_components.py
```

Do not create more folders unless truly necessary.

Do not create files such as:

```text
service.py
manager.py
repository.py
factory.py
base.py
abstract.py
```

This project does not need them.

---

# Responsibility of Each File

## app.py

This is the main Streamlit entry point.

It should only:

- configure the Streamlit page
- call sidebar UI
- call main prediction UI
- connect user actions to functions
- keep the overall flow readable

It should not contain heavy model logic.

---

## src/config.py

Contains project paths and fixed configuration.

Examples:

- project root
- model folders
- default threshold
- supported image types

---

## src/disease_info.py

Contains disease names and Vietnamese display names.

Examples:

- disease list
- disease translation map
- get Vietnamese disease name function

---

## src/model_loader.py

Contains model-related loading logic.

Examples:

- get_resnet50_model
- get_densenet121_model
- load_weights
- discover_diseases
- load CE model
- load DAM model
- load DenseNet121 model

This file should answer:

> How are models created and loaded?

---

## src/image_processing.py

Contains image preprocessing logic.

Examples:

- convert image to RGB
- resize to 224x224
- convert to tensor
- normalize using ImageNet mean/std

This file should answer:

> How is the input X-ray image prepared before being sent into the model?

---

## src/prediction.py

Contains prediction and result interpretation logic.

Examples:

- predict binary disease probability
- predict multi-label disease probabilities
- interpret probability using threshold
- compare CE and DAM probability

This file should answer:

> How does the system convert model output into understandable prediction results?

---

## src/gradcam.py

Contains Grad-CAM logic only.

Examples:

- GradCAM class
- get target layer
- generate CAM
- apply heatmap overlay
- generate Grad-CAM visualization

This file should answer:

> How does the system visualize the area that the model focuses on?

---

## src/ui_components.py

Contains reusable Streamlit display functions.

Examples:

- sidebar controls
- medical disclaimer
- display uploaded image
- display prediction table
- display probability metric
- display Grad-CAM images

This file should not contain model internals.

---

# Refactor Rules

## Rule 1 — Preserve behavior

Do not change the meaning of the program.

The following features must continue to work exactly as before:

- upload chest X-ray image
- choose ResNet50 DAM vs CE
- choose DenseNet121 multi-label
- choose disease
- adjust threshold
- run prediction
- show probability
- show interpretation
- show comparison chart
- show Grad-CAM
- show medical disclaimer

---

## Rule 2 — Student-level readability

Code should be readable by a university student.

Prefer simple function names.

Good examples:

```python
load_dam_model()
load_ce_model()
preprocess_image()
predict_single_disease()
predict_all_diseases()
display_prediction_result()
generate_gradcam_visualization()
```

Avoid vague names:

```python
process()
handle()
manager()
service()
execute()
```

---

## Rule 3 — Do not split too much

Only split code when the responsibility becomes clearer.

Do not create tiny files with only one or two trivial functions.

The goal is not "many files".

The goal is "easy to understand".

---

## Rule 4 — Do not rewrite working logic unnecessarily

Move code carefully.

Do not change formulas, preprocessing, thresholds, model output handling, or Grad-CAM logic unless there is an actual bug.

If a change might affect model output, explain it first and ask for approval.

---

## Rule 5 — Keep Streamlit simple

The Streamlit app should still be easy to follow.

Avoid complex callback systems.

Avoid class-based UI.

Avoid unnecessary abstraction.

This should still feel like a student Streamlit project.

---

# Digital Image Processing Focus

Because this is a Digital Image Processing project, the refactor must make these parts easy to explain:

- how the uploaded image is read
- why the image is converted to RGB
- why the image is resized to 224x224
- why ImageNet normalization is used
- how the tensor is passed into the model
- why sigmoid is used for binary / multi-label prediction
- why softmax may be used for 2-class CE output
- how threshold affects final interpretation
- how Grad-CAM creates a heatmap
- how CE and DAM models are compared

The code organization should support oral defense.

A lecturer should be able to ask:

> Where is image preprocessing?

and the student can immediately answer:

> src/image_processing.py

A lecturer should ask:

> Where is Grad-CAM?

and the student can immediately answer:

> src/gradcam.py

---

# Step-by-Step Work Plan

Do the refactor in phases.

## Phase 1 — Analyze Current Code

Before changing code, summarize:

- What the current main file is doing
- What model_utils.py is doing
- Which parts are mixed together
- Proposed final file responsibilities

Do not modify files yet.

---

## Phase 2 — Create New Files

Create the simple `src/` structure.

Move only clearly related functions.

Do not change behavior.

---

## Phase 3 — Update Imports

Update imports in `app.py`.

Prefer readable imports:

```python
from src.image_processing import preprocess_image
from src.prediction import predict_single_disease
from src.gradcam import generate_gradcam_visualization
```

Do not use wildcard imports.

Never use:

```python
from module import *
```

---

## Phase 4 — Verify App Launch

Run:

```bash
streamlit run app.py
```

or the correct command used by the project.

Fix import errors before continuing.

---

## Phase 5 — Verify Features

Verify each feature manually or logically:

- image upload works
- sidebar controls work
- ResNet50 CE/DAM mode works
- DenseNet121 multi-label mode works
- threshold interpretation works
- Grad-CAM works
- result tables still display
- charts still display
- medical disclaimer still displays

---

# Final Output Required

After refactoring, report clearly:

## 1. What changed

Explain which code moved where.

## 2. Why this is easier to understand

Explain using student-level language.

## 3. Final file responsibility summary

For each file, write one sentence:

Example:

```text
app.py: Runs the Streamlit application and connects the UI with prediction functions.
src/image_processing.py: Prepares uploaded X-ray images before model prediction.
src/gradcam.py: Generates heatmap visualizations showing where the model focuses.
```

## 4. Verification result

List checked features:

```text
[ ] App launches
[ ] Upload image works
[ ] ResNet50 CE/DAM works
[ ] DenseNet121 works
[ ] Grad-CAM works
[ ] Threshold works
```

Mark what passed and what still needs manual testing.

---

# Strict Restrictions

Do NOT:

- rewrite the whole project from scratch
- change model behavior
- change prediction formulas
- change preprocessing values
- remove medical disclaimer
- add unnecessary classes
- add enterprise architecture
- create too many files
- create wrapper-only files
- make the project harder to explain

The final project should feel like:

> a clean, understandable student project for a Digital Image Processing course.
