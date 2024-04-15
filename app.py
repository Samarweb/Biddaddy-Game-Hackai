import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window

# Load data
data = pd.read_csv('data.csv')

# Preprocess data
X = data['period_number'].apply(lambda x: int(str(x)[-3:])).values.reshape(-1, 1)
y = data['outcome'].map({'Big': 1, 'Small': 0}).values

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.label = Label(text='Enter period number:', color=(1, 0, 0, 1))  # Set label text color to red
        self.input = TextInput(multiline=False, background_color=(1, 0, 0, 1))  # Set text input background color to red
        self.button = Button(text='Predict', on_press=self.predict, background_color=(1, 0, 0, 1))  # Set button background color to red
        
        layout.add_widget(self.label)
        layout.add_widget(self.input)
        layout.add_widget(self.button)
        
        return layout
    
    def predict(self, instance):
        period_number = int(self.input.text)
        prediction = model.predict(np.array([[period_number]]))[0]
        result = 'Big' if prediction == 1 else 'Small'
        self.label.text = f'Result: {result}'

if __name__ == '__main__':
    MyApp().run()
