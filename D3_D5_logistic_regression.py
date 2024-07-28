import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Загрузка данных из Excel файла
df = pd.read_excel('data.xlsx', sheet_name='LSI')

# Определение вопросов для доменов D3 и D5
d3_questions = [2, 14, 18, 26, 33, 48, 50, 58, 69, 78, 86, 88, 93, 95]
d5_questions = [7, 9, 23, 27, 38, 41, 55, 63, 71, 73, 84, 92, 96]

# Создание списка всех вопросов из доменов D3 и D5
all_questions = d3_questions + d5_questions

# Подготовка данных
# Выделение вопросов и их ответы
df_questions = df[['Q' + str(q) for q in all_questions]]


# Функция для проведения логистической регрессии и оценки модели
def logistic_regression_analysis(question, df):
    # Зависимая переменная: ответы на конкретный вопрос (1 - "да", 0 - "нет")
    Y = df['Q' + str(question)]

    # Независимые переменные: ответы на все остальные вопросы из выбранных доменов
    X = df.drop(columns=['Q' + str(question)])

    # Разделение данных на тренировочные и тестовые
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Проверка распределения классов в тренировочных и тестовых данных
    #print(f"Распределение классов для вопроса Q{question} в исходных данных:\n{Y.value_counts()}")
    #print(f"Распределение классов для вопроса Q{question} в обучающей выборке:\n{Y_train.value_counts()}")
    #print(f"Распределение классов для вопроса Q{question} в тестовой выборке:\n{Y_test.value_counts()}")

    # Логистическая регрессия
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Предсказание на тестовых данных
    Y_pred = model.predict(X_test)

    # Оценка модели
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, zero_division=1)

    print(f"Результаты для вопроса Q{question}:")
    print(f"Точность (Accuracy): {accuracy}")
    print(report)


# Анализ для каждого вопроса в доменах D3 и D5
for question in all_questions:
    logistic_regression_analysis(question, df_questions)
