## Dataset Documentation: promisAI v1

### Overview

This dataset contains 306 annotated records derived from the archived projects of the **Forest Products Research and Development Institute (FPRDI)** or **REDEEM**. Each record links historical research to relevant **concepts**, the **Sustainable Development Goals (SDGs)**, the **8-point socioeconomic agenda of President Marcos**, and the **Research & Development Priority Area & Programs (HNRDA)**. The annotations aim to support further research, classification, and policy alignment studies.

> **Access**: This is a **private** dataset not intended for public distribution.

---

### Dataset Summary

| Feature    | Description                                          |
| ---------- | ---------------------------------------------------- |
| Rows       | 306                                                  |
| Columns    | 5                                                    |
| File Size  | 153 KB                                               |
| Format     | JSON                                                 |
| Version    | 1.0                                                  |
| Language   | English                                              |
| Tools Used | Python (Pandas, JSON libraries)                      |
| Annotation | GPT API-based pipeline (abstract context extraction) |

---

### Column Descriptions

| Column Name                     | Description                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------- |
| `title`                         | The original, unedited project title from the FPRDI archive.                 |
| `concept`                       | A list of keywords representing the core research concept of the project.    |
| `sdg`                           | Related **Sustainable Development Goals** aligned with the project's impact. |
| `presidential_development_plan` | Tagged based on the **8-Point Socioeconomic Agenda** of President Marcos Jr. |
| `hnrda`                         | Tags from the **Research & Development Priority Area & Programs**            |

---

### Extraction

The dataset was compiled through a hybrid extraction approach combining OCR, PDF parsing, and rule-based logic:

- **Tesseract OCR**  
  Used to extract text from scanned documents and image-based PDFs, enabling retrieval of data from non-selectable text sources.

- **MuPDF Library**  
  Employed for parsing and interpreting the layout of PDF files, allowing for precise access to structured and semi-structured content.

- **Regex-based Rule Engine**  
  Developed to identify and isolate key components such as **titles** and **abstracts** using pattern matching techniques tailored to the FPRDI archive format.

This multi-step pipeline ensured accurate, consistent, and reliable extraction of research data from a variety of historical document types.

---

### Annotation Methodology

* **Pipeline Used**: The dataset was annotated using an automated **GPT API pipeline**.
* **Basis for Tagging**: Abstract-based context extraction to infer SDG, 8Ps, and HNRDA tags.

* **Post-Processing**: Cleaned and validated using Python for consistency and accuracy.

---

### Tagging Frameworks

* **SDGs**: Based on the UN’s 17 Sustainable Development Goals.
* **Presidential Development Plan**: Based on President Bongbong Marcos Jr.’s 8-Point Socioeconomic Agenda.
* **HNRDA**: Based on Research & Development Priority Area & Programs

---
Great — here's the finalized **Model Training Results** section with your provided metrics and details properly formatted:

---

### Model Training Results

This dataset was used to train an AI model for **title generation** based on structured inputs. The goal is to automatically generate concise and relevant research titles using the provided concepts, SDG alignment, Presidential Development Plan (8Ps), and HNRDA classifications.

#### Training Setup

| Aspect             | Details                                                 |
| ------------------ | ------------------------------------------------------- |
| Model              | FLAN-T5-base                                            |
| Framework          | Hugging Face Transformers                               |
| Task               | Text Generation (title generation)                      |
| Input Format       | `"title"` + `"concept"` + `"sdg"` + `"8Ps"` + `"hnrda"` |
| Evaluation Metrics | ROUGE-L, BERTScore, Average Cosine Similarity           |

#### Evaluation Results

Of course! Here’s an explanation in simpler terms:

---

### Evaluation Results

| Metric                    | Score  |
| ------------------------- | ------ |
| ROUGE-L                   | 0.5833 |
| BERTScore (F1)            | 0.8913 |
| Average Cosine Similarity | 0.8998 |

#### What These Metrics Mean

* **ROUGE-L (0.5833)**:
  This score checks how much of the **same words or phrases** are used in the generated title compared to the original. A score of **0.58** means that the generated title has a **good overlap** with the original, but it's not exactly the same. In tasks like generating titles, it's okay to have some differences, as long as the meaning is still captured.

* **BERTScore (0.8913)**:
  This score checks how **similar the meaning** of the generated title is to the original one, even if the words are different. A score of **0.89** is **really good**, meaning the model is **very close** to the meaning of the original title, even if it uses different words.

* **Average Cosine Similarity (0.8998)**:
  This score compares how **closely related the ideas** in the generated title are to the original. A score of **0.9** means the model is doing a **great job** in making sure the generated title has a **similar meaning** to the original one.

> **In Simple Terms**: The model is **doing a good job** of generating titles that are **meaningfully similar** to the original, even if the exact words aren't the same. These numbers show that the model can **understand** and **reproduce** the main idea well, which is what matters most for title generation.