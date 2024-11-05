import numpy as np
import pandas as pd
import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from tensorflow.keras.utils import to_categorical

# Configurações
confidence_threshold = 0.95
batch_size = 32
initial_epochs = 10
pseudo_epochs = 5
num_classes = 6
csv_path = "experiment_metrics.csv"

# Inicializar dataframe para armazenar métricas
metrics_df = pd.DataFrame(columns=[
    'Tempo', 'test_loss', 'test_accuracy', 'precision', 'recall', 'fscore', 'kappa',
    'str_time', 'end_time', 'duration', 'best_epoch', 'ini_label', 'select_pseudo',
    'rest_unlabels', 'total_training_base', 'new_train', 'id_test'
])

# Carregar os caminhos das imagens e rótulos de arquivos CSV
df = pd.read_csv("caminho_para_arquivo_rotulado.csv")
df_unlabeled = pd.read_csv("caminho_para_arquivo_nao_rotulado.csv")
df_test = pd.read_csv("caminho_para_arquivo_teste.csv")

# Dividir os dados em dados rotulados e não rotulados
X_labeled = df['image_path'].values
y_labeled = df['label'].values
X_unlabeled = df_unlabeled['image_path'].values
y_labeled = to_categorical(y_labeled, num_classes=num_classes)

# Função para calcular e salvar métricas
def log_metrics(tempo, test_data, model, start_time, end_time, ini_label, select_pseudo, rest_unlabels, total_training_base, new_train):
    test_loss, test_accuracy = model.evaluate(test_data)
    y_true = test_data.classes
    y_pred = np.argmax(model.predict(test_data), axis=1)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    duration = end_time - start_time
    best_epoch = early_stopping.stopped_epoch - early_stopping.patience
    
    metrics_df.loc[len(metrics_df)] = [
        tempo, test_loss, test_accuracy, precision, recall, fscore, kappa,
        time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time)),
        time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time)),
        duration, best_epoch, ini_label, select_pseudo, rest_unlabels,
        total_training_base, new_train, "id_teste"
    ]
    metrics_df.to_csv(csv_path, index=False)

# Pré-processamento dos dados
data_gen = ImageDataGenerator(rescale=1./255)

# Função para criar geradores de dados
def create_data_generator(image_paths, labels=None):
    if labels is not None:
        return data_gen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': image_paths, 'class': labels}),
            x_col='filename',
            y_col='class',
            target_size=(224, 224),
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True
        )
    else:
        return data_gen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': image_paths}),
            x_col='filename',
            target_size=(224, 224),
            class_mode=None,
            batch_size=batch_size,
            shuffle=False
        )

# Criar geradores de dados
train_data_labeled = create_data_generator(X_labeled, y_labeled)
train_data_unlabeled = create_data_generator(X_unlabeled)
test_data = create_data_generator(df_test['image_path'].values, to_categorical(df_test['label'].values, num_classes=num_classes))

# Carregar o modelo pré-treinado (ResNet50) e ajustar para o nosso problema
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Descongelar todas as camadas para DFT
for layer in model.layers:
    layer.trainable = True

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar callbacks para monitoramento e checkpoint
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, min_delta=1e-4)
model_checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True, save_weights_only=True)

# Treinar o modelo com os dados rotulados (tempo 0)
tempo = 0
start_time = time.time()
model.fit(train_data_labeled, epochs=initial_epochs, callbacks=[early_stopping, model_checkpoint])
end_time = time.time()

# Registrar métricas para o tempo 0
log_metrics(tempo, test_data, model, start_time, end_time, len(X_labeled), 0, len(X_unlabeled), len(X_labeled), 0)

# Loop de pseudo-rotulação
while len(X_unlabeled) > 0:
    tempo += 1
    start_time = time.time()
    
    # Fazer previsões na base não rotulada
    predictions = model.predict(train_data_unlabeled)
    pseudo_labels = np.argmax(predictions, axis=1)
    pseudo_confidences = np.max(predictions, axis=1)

    # Selecionar pseudo-rótulos com alta confiança
    high_confidence_indices = np.where(pseudo_confidences >= confidence_threshold)[0]
    if len(high_confidence_indices) == 0:
        print("Nenhum pseudo-rótulo de alta confiança restante. Parando.")
        break

    # Atualizar as bases rotuladas e não rotuladas
    X_pseudo = X_unlabeled[high_confidence_indices]
    y_pseudo = to_categorical(pseudo_labels[high_confidence_indices], num_classes=num_classes)
    X_labeled = np.concatenate([X_labeled, X_pseudo])
    y_labeled = np.concatenate([y_labeled, y_pseudo])
    
    # Remover os dados pseudo-rotulados da base não rotulada
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[high_confidence_indices] = False
    X_unlabeled = X_unlabeled[mask]

    # Re-criar o gerador de dados rotulados com a nova base
    train_data_labeled = create_data_generator(X_labeled, y_labeled)

    # Re-treinar o modelo com a base atualizada
    model.fit(train_data_labeled, epochs=pseudo_epochs, callbacks=[early_stopping, model_checkpoint])
    end_time = time.time()
    
    # Registrar métricas para o tempo atual
    log_metrics(tempo, test_data, model, start_time, end_time, len(X_labeled) - len(high_confidence_indices), len(high_confidence_indices), len(X_unlabeled), len(X_labeled), len(high_confidence_indices))

print("Treinamento com pseudo-rotulação concluído.")
