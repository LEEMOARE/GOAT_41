# GOAT_41

<div align="center">

## 📌&nbsp;&nbsp;The overall flowchart of the project.

```mermaid

flowchart TD
A[to be updated]

```

<br>

---

<br>

```mermaid

---
title: [AI-Model flow chart]
---
flowchart TD

    A[Portable X-ray] 
    B{Model}
    C{Detected?}
    D[Overlay lesion]
    E[result]
    
    A -- " pre-processing " --> B
    B -- " prediction " --> C
    C -- "yes" --> D 
    D --> E
    C -- "no" --> E
```
</div>

## 1. Usage

### 1.1 how to train? 

```python
python3 train.py tbu
```

### 1.2 how to evaluation?

```python
python3 evaluate.py tbu
```