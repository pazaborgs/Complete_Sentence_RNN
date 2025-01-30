import streamlit as st
import gdown

def load_model():

    url = '{url modificada}'
    gdown.download(url, 'modelo_vidente.keras')
    loaded_model = tf.keras.models.load_model('modelo_vidente.keras')

    with open('vectorizer.pkl', 'rb') as file:
    	vectorizer = pickle.load(file)
            
    return loaded_model, vectorizer

def predict_next_words(model, vectorizer, text, max_sequence_len, top_k=3):
    """
    Prediz as próximas palavras mais prováveis em uma sequência de texto.

    Args:
        model: O modelo treinado.
        vectorizer: A camada de vetorização.
        text: O texto de entrada.
        max_sequence_len: O comprimento máximo da sequência usado na vetorização.
        top_k: O número de palavras mais prováveis a serem retornadas.

    Returns:
        As próximas palavras mais prováveis.
    """

    # Vetorizar o texto de entrada
    tokenized_text = vectorizer([text])

    # Remover a dimensão extra adicionada pela vetorização
    tokenized_text = np.squeeze(tokenized_text)

    # Adicionar padding à esquerda
    padded_text = pad_sequences([tokenized_text], maxlen=max_sequence_len, padding='pre')

    # Fazer a previsão
    predicted_probs = model.predict(padded_text, verbose=0)[0]  # Remove a dimensão extra adicionada pela previsão

    # Obter os índices dos top_k tokens com as maiores probabilidades
    top_k_indices = np.argsort(predicted_probs)[-top_k:][::-1]

    # Converter os tokens previstos de volta para palavras
    predicted_words = [vectorizer.get_vocabulary()[index] for index in top_k_indices]

    return predicted_words


def main():

    max_sequence_len = 50

    # carregar_modelo

    st.title('Previsão de Próximas Palavras')

    input_text = st.text_input('Digite uma sequência de texto:')

    load_model, vectorizer = load_model()

if st.button('Prever'):
    if input_text:
        try:
            predicted_words = predict_next_words(loaded_model, vectorizer, input_text, max_sequence_len)
            
            st.info('Palavras mais prováveis')
            
            for word in predicted_words:
                st.success(word)
        except:
            st.error('Erro na previsão {e}')
    else:
        st.warning('Por favor, insira algum texto')

# código omitido

        
if __name__ == "__main__":
     main()

