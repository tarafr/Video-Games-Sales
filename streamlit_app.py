import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
from apps import home, prediction

# Конфигурация страницы Streamlit
st.set_page_config(
    page_title="Анализ продаж видеоигр",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, encoding = 'Latin-1')
    # Удаление пропусков
    df.dropna(inplace=True)

    # Преобразование столбцов к категориальным типам
    categorical_columns = [
        'Platform', 'Genre', 'Publisher',
    ]
    for column in categorical_columns:
        df[column] = df[column].astype('category')

    df['Year'] = df['Year'].astype(int)
    return df


class Menu:
    apps = [
        {
            "func": home.app,
            "title": "Главная",
            "icon": "house-fill"
        },
        {
            "func": prediction.app,
            "title": "Прогнозирование",
            "icon": 'graph-up'
        },
    ]

    def run(self):
        with st.sidebar:
            titles = [app["title"] for app in self.apps]
            icons = [app["icon"] for app in self.apps]
            st.image('images/logo.webp', use_column_width='auto')

            selected = option_menu(
                "Меню",
                options=titles,
                icons=icons,
                menu_icon="cast",
                default_index=0,
            )

            st.info("""
                ### Область применения
                Система анализа данных помогает разработчикам, издателям и аналитикам в игровой индустрии понимать текущие тенденции и предпочтения потребителей. 
                Это инструмент для изучения популярности различных игр, платформ, жанров, а также для анализа влияния года выпуска и издателя на продажи видеоигр.
                Система позволяет исследовать региональные предпочтения и выявлять ключевые факторы, влияющие на коммерческий успех игр, что способствует 
                принятию обоснованных решений в процессе разработки и маркетинга видеоигр.
            """)
        return selected


if __name__ == '__main__':
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    df = load_data(current_dir / 'vgsales.csv')

    menu = Menu()
    selected = menu.run()

    # Добавление интерфейса для загрузки файла
    st.sidebar.header('Загрузите свой файл')
    uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif df is None:
        st.sidebar.warning("Пожалуйста, загрузите файл данных.")

    for app in menu.apps:
        if app["title"] == selected:
            app["func"](df, current_dir)
            break
