# nlp-sample
# **Шпаргалка по NLP (Natural Language Processing) для новичков**

---

## **0. Что такое NLP?**
NLP — это обработка естественного языка: учим компьютер понимать и работать с текстом.

**Примеры задач NLP:**
- Классификация текста (спам/не спам, позитив/негатив)
- Перевод текста
- Генерация текста
- Ответ на вопросы
- Чат-боты

---

## **1. Что загружать первым?**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```

---

## **2. Как выглядит базовый пайплайн NLP задачи:**

```python
# 1. Загружаем данные
df = pd.read_csv("your_text_data.csv")

# 2. Смотрим пример текста
print(df.head())

# 3. Разделение
X = df["text"]         # Столбец с текстами
y = df["label"]        # Метки (например, 0 — негатив, 1 — позитив)

# 4. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Векторизация текста
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Обучение модели
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Оценка
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## **3. Если задача такая... — то модель такая:**

| Задача                        | Что использовать                         |
|------------------------------|------------------------------------------|
| Спам/не спам                 | `TfidfVectorizer` + `NaiveBayes`         |
| Анализ тональности (позитив/негатив) | `LogisticRegression` или `SVM`         |
| Классификация новостей       | `TfidfVectorizer` + `RandomForest`       |
| Генерация текста             | `GPT-2`, `GPT-3`                         |
| Ответы на вопросы            | `BERT`, `T5`, `DistilBERT`               |

---

## **4. Предобработка текста (если нужно вручную):**

```python
import re

def clean_text(text):
    text = text.lower()  # нижний регистр
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # убрать спецсимволы
    text = re.sub(r"\s+", " ", text).strip()  # убрать лишние пробелы
    return text

df["text"] = df["text"].apply(clean_text)
```

---

## **5. Hugging Face Transformers (простой ввод)**

```python
from transformers import pipeline

# Классификация текста (эмоции, тональность)
classifier = pipeline("sentiment-analysis")
print(classifier("I love machine learning!"))
```

---

## **6. Советы для задач на олимпиаде:**

- Для простых задач (спам/эмоции) — `Tfidf + Naive Bayes` или `LogReg`
- Для продвинутых (перевод, QA) — `transformers` (BERT, T5)
- Всегда делай: чистка текста, токенизация, векторизация
- Если запрещено использовать токенизаторы — НЕ используй `BERT`, а используй `Tfidf`
- Используй `.str.lower()`, `.str.replace()` для предобработки

---

## **7. Где брать датасеты для практики:**
- [Kaggle: NLP Datasets](https://www.kaggle.com/datasets?search=nlp)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

---

## **8. Где учиться и смотреть код:**
- https://huggingface.co/learn/nlp-course/
- https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- https://www.kaggle.com/learn/natural-language-processing

---

**Главное: не бойся текста. Он просто строка — ты сильнее!**
