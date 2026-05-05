import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTETomek

# -----------------------------------------------------------------
# 1. Clase
# -----------------------------------------------------------------
class ClassifierWithImbalanceClass:
    def __init__(self):
        """
        Constructor de la clase.
        Pipeline de scikit-learn.
        """
        
        # Estrategia:
        # 1. StandardScaler: SVMs son sensibles a la escala.
        # 2. SMOTE: Soluciona el desbalance masivo (el hint).
        # 3. LinearSVC: El modelo requerido.
        
        self._pipeline = make_pipeline(
            RobustScaler(),
            #StandardScaler(),
            SMOTETomek(random_state=42),
            #SMOTE(random_state=42),
            #SMOTEENN(random_state=42),
            LinearSVC(random_state=42)
        )

    def train(self, x, y):
        """
        Entrena el clasificador (pipeline completo).
        """
        # Guardamos el pipeline ajustado en self.classifier
        self.classifier = self._pipeline.fit(x, y)

    def predict(self, x):
        """
        Predice las clases para nuevas muestras.
        """
        # Llamamos al método .predict() del pipeline ya entrenado
        return self.classifier.predict(x)

# -----------------------------------------------------------------
# 2. EL SCRIPT DE PRUEBA
# -----------------------------------------------------------------
def main():
    """
    Función principal para cargar datos y probar el clasificador.
    """
    print("Cargando datos...")
    # Los CSV están en la misma carpeta que este script
    try:
        df_train = pd.read_csv("./train_data.csv")
        y_train = df_train["target"]
        X_train = df_train.drop("target", axis=1)

        df_test = pd.read_csv("./test_data.csv")
        y_test = df_test["target"]
        X_test = df_test.drop("target", axis=1)
        
        print("Datos cargados exitosamente.")
    except FileNotFoundError:
        print("-" * 40)
        print("ERROR: No se encontraron los archivos CSV.")
        print("Asegúrate de que 'train_data.csv' y 'test_data.csv' estén")
        print("en la misma carpeta que este script de Python.")
        print("-" * 40)
        return

    print("Entrenando el clasificador...")
    # --- Usamos la clase ---
    classifier = ClassifierWithImbalanceClass()
    classifier.train(X_train, y_train)
    
    print("Realizando predicciones...")
    y_pred = classifier.predict(X_test)
    # -----------------------------

    print("\n--- Resultados de la evaluación ---")
    print(f"Tipo de y_pred:   {type(y_pred)}")
    print(f"Shape de y_pred:  {y_pred.shape}")
    
    # Calcular el score
    score = roc_auc_score(y_test, y_pred)
    
    print(f"\nScore ROC AUC: {score}")
    print("\n(El score base era ~0.61. ¡Cualquier valor más alto es una victoria!)")

# Esto hace que el script se ejecute cuando desde la terminal, pero no cuando se importe como módulo
if __name__ == "__main__":
    main()