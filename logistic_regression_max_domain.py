import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Загрузка данных из Excel файла (страница "LSI")
df = pd.read_excel('data.xlsx', sheet_name='LSI')

# Вывод названий столбцов для проверки
print("Названия столбцов в DataFrame:", df.columns)

# Предположим, что у вас есть столбцы D1, D2, ..., D8, представляющие сумму ответов "да" по каждому домену
domain_columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']

# Создаем новый столбец, который будет содержать домен с наибольшим числом ответов "да"
df['Max_Domain'] = df[domain_columns].idxmax(axis=1)

# Зависимая переменная: домен с наибольшим числом ответов "да"
Y = df['Max_Domain']

# Независимые переменные: ответы на все вопросы
# Проверка правильности названий столбцов
question_columns = [f'Q{i}' for i in range(1, 98)]  # "Q1", "Q2", ..., "Q97"
print("Ожидаемые названия столбцов для вопросов:", question_columns)

# Проверка наличия ожидаемых столбцов в DataFrame
missing_columns = [col for col in question_columns if col not in df.columns]
if missing_columns:
    print("Отсутствующие столбцы:", missing_columns)
else:
    X = df[question_columns]

    # Преобразование категориальных данных в числовые
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    # Разделение данных на тренировочные и тестовые
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

    # Логистическая регрессия
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Предсказание на тестовых данных
    Y_pred = model.predict(X_test)

    # Оценка модели
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_, zero_division=1)

    print("Accuracy:", accuracy)
    print(report)
