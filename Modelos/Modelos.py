import os
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict

# Obtener la ruta del directorio actual (Modelos)
MODELOS_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar los modelos entrenados desde archivos .pkl
print("Cargando modelos entrenados...")

try:
    content_model = joblib.load(os.path.join(MODELOS_DIR, 'content_model.pkl'))
    print("✓ Modelo de contenido cargado")
except Exception as e:
    print(f"⚠️ Error cargando modelo de contenido: {e}")
    content_model = None

try:
    copurchase_model = joblib.load(os.path.join(MODELOS_DIR, 'co_purchase_model.pkl'))
    print("✓ Modelo de co-compra cargado")
except Exception as e:
    print(f"⚠️ Error cargando modelo de co-compra: {e}")
    copurchase_model = None

try:
    coquote_model = joblib.load(os.path.join(MODELOS_DIR, 'co_quotation_model.pkl'))
    print("✓ Modelo de co-cotización cargado")
except Exception as e:
    print(f"⚠️ Error cargando modelo de co-cotización: {e}")
    coquote_model = None

try:
    hybrid_model = joblib.load(os.path.join(MODELOS_DIR, 'hybrid_model.pkl'))
    print("✓ Modelo híbrido cargado")
except Exception as e:
    print(f"⚠️ Error cargando modelo híbrido: {e}")
    hybrid_model = None

try:
    historical_model = joblib.load(os.path.join(MODELOS_DIR, 'historical_model.pkl'))
    print("✓ Modelo histórico cargado")
except Exception as e:
    print(f"⚠️ Error cargando modelo histórico: {e}")
    historical_model = None

print("Todos los modelos han sido cargados.\n")

# ================================================================
# FUNCIONES DE RECOMENDACIÓN PARA USAR LOS MODELOS CARGADOS
# ================================================================

def get_content_recommendations(input_product, N=10):
    """
    Genera recomendaciones basadas en contenido usando el modelo cargado.
    
    Args:
        input_product (str): ID del producto
        N (int): Número de recomendaciones
    
    Returns:
        pandas.DataFrame: Recomendaciones con columnas ['producto', 'similarity_score']
    """
    if content_model is None:
        print("Error: Modelo de contenido no disponible")
        return pd.DataFrame()
    
    similarity_matrix = content_model['similarity_matrix']
    product_to_idx = content_model['product_to_idx']
    idx_to_product = content_model['idx_to_product']
    
    if input_product not in product_to_idx:
        print(f"Error: Producto '{input_product}' no encontrado")
        return pd.DataFrame()
    
    idx = product_to_idx[input_product]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, score in sim_scores:
        if i == idx:
            continue
        product_name = idx_to_product.get(i)
        if product_name and len(recommendations) < N:
            recommendations.append({'producto': product_name, 'similarity_score': score})
        elif len(recommendations) >= N:
            break
    
    return pd.DataFrame(recommendations)

def get_copurchase_recommendations(input_product, N=10):
    """
    Genera recomendaciones basadas en co-compra usando el modelo cargado.
    
    Args:
        input_product (str): ID del producto
        N (int): Número de recomendaciones
    
    Returns:
        pandas.DataFrame: Recomendaciones con columnas ['producto', 'co_purchase_count']
    """
    if copurchase_model is None:
        print("Error: Modelo de co-compra no disponible")
        return pd.DataFrame()
    
    co_occurrence_matrix = copurchase_model['co_occurrence_matrix']
    product_to_idx = copurchase_model['product_to_idx']
    idx_to_product = copurchase_model['idx_to_product']
    
    if input_product not in product_to_idx:
        print(f"Error: Producto '{input_product}' no encontrado")
        return pd.DataFrame()
    
    idx = product_to_idx[input_product]
    co_buys = co_occurrence_matrix[idx, :]
    co_buy_indices = co_buys.indices
    co_buy_values = co_buys.data
    
    if len(co_buy_indices) == 0:
        print(f"Producto '{input_product}' no tiene co-compras registradas")
        return pd.DataFrame()
    
    recommendations = []
    cobuy_pairs = sorted(zip(co_buy_indices, co_buy_values), key=lambda x: x[1], reverse=True)
    
    for i, count in cobuy_pairs:
        if i == idx:
            continue
        product_name = idx_to_product.get(i)
        if product_name and len(recommendations) < N:
            recommendations.append({'producto': product_name, 'co_purchase_count': count})
        elif len(recommendations) >= N:
            break
    
    return pd.DataFrame(recommendations)

def get_coquotation_recommendations(input_product, N=10):
    """
    Genera recomendaciones basadas en co-cotización usando el modelo cargado.
    
    Args:
        input_product (str): ID del producto
        N (int): Número de recomendaciones
    
    Returns:
        pandas.DataFrame: Recomendaciones con columnas ['producto', 'co_quotation_count']
    """
    if coquote_model is None:
        print("Error: Modelo de co-cotización no disponible")
        return pd.DataFrame()
    
    co_quotation_matrix = coquote_model['co_quotation_matrix']
    product_to_idx = coquote_model['product_to_idx']
    idx_to_product = coquote_model['idx_to_product']
    
    if input_product not in product_to_idx:
        print(f"Error: Producto '{input_product}' no encontrado")
        return pd.DataFrame()
    
    idx = product_to_idx[input_product]
    co_quotes = co_quotation_matrix[idx, :]
    co_quote_indices = co_quotes.indices
    co_quote_values = co_quotes.data
    
    if len(co_quote_indices) == 0:
        print(f"Producto '{input_product}' no tiene co-cotizaciones registradas")
        return pd.DataFrame()
    
    recommendations = []
    coquote_pairs = sorted(zip(co_quote_indices, co_quote_values), key=lambda x: x[1], reverse=True)
    
    for i, count in coquote_pairs:
        if i == idx:
            continue
        product_name = idx_to_product.get(i)
        if product_name and len(recommendations) < N:
            recommendations.append({'producto': product_name, 'co_quotation_count': count})
        elif len(recommendations) >= N:
            break
    
    return pd.DataFrame(recommendations)

def get_hybrid_recommendations(input_product, N=10, content_weight=0.3, cf_buy_weight=0.5, cf_quote_weight=0.2, k=2):
    """
    Genera recomendaciones híbridas usando re-ranking.
    
    Args:
        input_product (str): ID del producto
        N (int): Número de recomendaciones finales
        content_weight (float): Peso para contenido
        cf_buy_weight (float): Peso para co-compra
        cf_quote_weight (float): Peso para co-cotización
        k (int): Constante de suavizado
    
    Returns:
        pandas.DataFrame: Recomendaciones con columnas ['producto', 'hybrid_score']
    """
    if hybrid_model is None:
        print("Error: Modelo híbrido no disponible")
        return pd.DataFrame()
    
    # Obtener candidatos de cada método
    content_recs = get_content_recommendations(input_product, N=50)
    copurchase_recs = get_copurchase_recommendations(input_product, N=50)
    coquote_recs = get_coquotation_recommendations(input_product, N=50)
    
    # Lógica adaptativa de pesos
    if not copurchase_recs.empty and copurchase_recs.iloc[0]['co_purchase_count'] < 10:
        content_weight = 0.3
        cf_buy_weight = 0.5
        cf_quote_weight = 0.2
    
    # Calcular scores híbridos
    hybrid_scores = defaultdict(float)
    
    # Contribución del modelo de contenido
    if not content_recs.empty:
        for rank, row in enumerate(content_recs.itertuples(), 1):
            score_contribution = content_weight * (1.0 / (rank + k))
            hybrid_scores[row.producto] += score_contribution
    
    # Contribución del modelo de co-compra
    if not copurchase_recs.empty:
        for rank, row in enumerate(copurchase_recs.itertuples(), 1):
            score_contribution = cf_buy_weight * (1.0 / (rank + k))
            hybrid_scores[row.producto] += score_contribution
    
    # Contribución del modelo de co-cotización
    if not coquote_recs.empty:
        for rank, row in enumerate(coquote_recs.itertuples(), 1):
            score_contribution = cf_quote_weight * (1.0 / (rank + k))
            hybrid_scores[row.producto] += score_contribution
    
    if not hybrid_scores:
        print(f"No se encontraron candidatos para '{input_product}'")
        return pd.DataFrame()
    
    # Crear DataFrame final
    final_recs = pd.DataFrame(hybrid_scores.items(), columns=['producto', 'hybrid_score'])
    final_recs = final_recs[final_recs['producto'] != input_product]
    final_recs = final_recs.sort_values('hybrid_score', ascending=False).head(N)
    
    return final_recs

def get_historical_recommendations(cliente_id, N=10, k_similar_items=30):
    """
    Genera recomendaciones personalizadas basadas en el historial del cliente usando el modelo cargado.
    
    Args:
        cliente_id (int): ID del cliente
        N (int): Número de recomendaciones
        k_similar_items (int): Número de productos similares a considerar por cada producto comprado
    
    Returns:
        pandas.DataFrame: Recomendaciones con columnas ['producto', 'recommendation_score']
    """
    if historical_model is None:
        print("Error: Modelo histórico no disponible")
        return pd.DataFrame()
    
    # Extraer componentes del modelo
    user_item_matrix_csr = historical_model['user_item_matrix_csr']
    item_similarity_matrix = historical_model['item_similarity_matrix']
    user_to_idx = historical_model['user_to_idx']
    idx_to_item = historical_model['idx_to_item']
    
    # Validaciones
    if not historical_model['data_ready']:
        print("Error: Modelo histórico no está listo")
        return pd.DataFrame()
    
    if cliente_id not in user_to_idx:
        print(f"Error: Cliente {cliente_id} no encontrado en el modelo")
        return pd.DataFrame()
    
    try:
        # Obtener índice del usuario
        target_user_idx = user_to_idx[cliente_id]
        user_purchased_items_indices = user_item_matrix_csr[target_user_idx].indices
        
        if len(user_purchased_items_indices) == 0:
            print(f"Cliente {cliente_id} no tiene historial de compras")
            return pd.DataFrame()
        
        # Acumular puntajes de candidatos
        candidate_items = defaultdict(float)
        
        for purchased_item_idx in user_purchased_items_indices:
            if purchased_item_idx >= item_similarity_matrix.shape[0]:
                continue
            
            # Obtener similitudes de este producto comprado
            item_similarities_vector = item_similarity_matrix[purchased_item_idx]
            similar_item_indices_sorted = np.argsort(item_similarities_vector)[::-1]
            top_k_indices = similar_item_indices_sorted[:k_similar_items]
            
            # Acumular puntajes de productos similares
            for similar_item_idx in top_k_indices:
                similarity_score = item_similarities_vector[similar_item_idx]
                if similarity_score > 0:
                    candidate_items[similar_item_idx] += similarity_score
        
        if not candidate_items:
            print(f"No se encontraron productos similares para cliente {cliente_id}")
            return pd.DataFrame()
        
        # Crear ranking de recomendaciones
        ranked_candidates = []
        for item_idx, score in candidate_items.items():
            item_id = idx_to_item.get(item_idx)
            if item_id:
                ranked_candidates.append({
                    'producto': item_id, 
                    'recommendation_score': float(score)
                })
        
        if not ranked_candidates:
            print(f"No se pudieron mapear productos para cliente {cliente_id}")
            return pd.DataFrame()
        
        # Crear DataFrame y ordenar
        recommendations = pd.DataFrame(ranked_candidates)
        recommendations = recommendations.sort_values('recommendation_score', ascending=False).head(N)
        return recommendations
        
    except Exception as e:
        print(f"Error generando recomendaciones históricas para cliente {cliente_id}: {e}")
        return pd.DataFrame()

# ================================================================
# FUNCIÓN DE PRUEBA PARA TODOS LOS MODELOS
# ================================================================

def test_all_models(test_product="producto_125", N=5):
    """
    Prueba todos los modelos con un producto de ejemplo y devuelve los resultados.
    
    Args:
        test_product (str): Producto para probar
        N (int): Número de recomendaciones por modelo
    
    Returns:
        dict: Diccionario con las recomendaciones de los 4 modelos:
              {
                'content': DataFrame,
                'co_purchase': DataFrame, 
                'co_quotation': DataFrame,
                'hybrid': DataFrame,
                'input_product': str
              }
    """
    print(f"🧪 PROBANDO TODOS LOS MODELOS CON: {test_product}")
    print("=" * 60)
    
    # Inicializar diccionario de resultados
    results = {
        'input_product': test_product,
        'content': pd.DataFrame(),
        'co_purchase': pd.DataFrame(),
        'co_quotation': pd.DataFrame(),
        'hybrid': pd.DataFrame()
    }
    
    # Probar modelo de contenido
    print("\n📊 MODELO DE CONTENIDO:")
    content_recs = get_content_recommendations(test_product, N=N)
    results['content'] = content_recs
    if not content_recs.empty:
        print(content_recs)
    else:
        print("No se encontraron recomendaciones de contenido")
    
    # Probar modelo de co-compra
    print("\n🛒 MODELO DE CO-COMPRA:")
    copurchase_recs = get_copurchase_recommendations(test_product, N=N)
    results['co_purchase'] = copurchase_recs
    if not copurchase_recs.empty:
        print(copurchase_recs)
    else:
        print("No se encontraron recomendaciones de co-compra")
    
    # Probar modelo de co-cotización
    print("\n💰 MODELO DE CO-COTIZACIÓN:")
    coquote_recs = get_coquotation_recommendations(test_product, N=N)
    results['co_quotation'] = coquote_recs
    if not coquote_recs.empty:
        print(coquote_recs)
    else:
        print("No se encontraron recomendaciones de co-cotización")
    
    # Probar modelo híbrido
    print("\n🎯 MODELO HÍBRIDO:")
    hybrid_recs = get_hybrid_recommendations(test_product, N=N)
    results['hybrid'] = hybrid_recs
    if not hybrid_recs.empty:
        print(hybrid_recs)
    else:
        print("No se encontraron recomendaciones híbridas")
    
    print("\n" + "=" * 60)
    print("✅ Prueba completada!")
    
    # Resumen de productos únicos encontrados
    all_products = set()
    for model_name, df in results.items():
        if model_name != 'input_product' and not df.empty:
            all_products.update(df['producto'].tolist())
    
    print(f"\n📈 RESUMEN:")
    print(f"- Total de productos únicos recomendados: {len(all_products)}")
    print(f"- Contenido: {len(results['content'])} productos")
    print(f"- Co-compra: {len(results['co_purchase'])} productos") 
    print(f"- Co-cotización: {len(results['co_quotation'])} productos")
    print(f"- Híbrido: {len(results['hybrid'])} productos")
    
    return results

def test_historical_model(test_cliente_id=10, N=10):
    """
    Prueba el modelo histórico con un cliente de ejemplo.
    
    Args:
        test_cliente_id (int): Cliente para probar
        N (int): Número de recomendaciones
    
    Returns:
        dict: Diccionario con las recomendaciones del modelo histórico:
              {
                'historical': DataFrame,
                'input_cliente': int
              }
    """
    print(f"🧪 PROBANDO MODELO HISTÓRICO CON CLIENTE: {test_cliente_id}")
    print("=" * 60)
    
    # Inicializar diccionario de resultados
    results = {
        'input_cliente': test_cliente_id,
        'historical': pd.DataFrame()
    }
    
    # Probar modelo histórico
    print("\n👤 MODELO HISTÓRICO (RECOMENDACIÓN PERSONALIZADA):")
    historical_recs = get_historical_recommendations(test_cliente_id, N=N)
    results['historical'] = historical_recs
    
    if not historical_recs.empty:
        print(historical_recs)
        print(f"\n📊 Información adicional:")
        print(f"- Cliente evaluado: {test_cliente_id}")
        print(f"- Recomendaciones generadas: {len(historical_recs)}")
        print(f"- Puntaje máximo: {historical_recs['recommendation_score'].max():.4f}")
        print(f"- Puntaje mínimo: {historical_recs['recommendation_score'].min():.4f}")
    else:
        print("No se encontraron recomendaciones históricas")
    
    print("\n" + "=" * 60)
    print("✅ Prueba del modelo histórico completada!")
    
    return results

# Exportar todo
__all__ = [
    'content_model', 
    'copurchase_model', 
    'coquote_model', 
    'hybrid_model',
    'historical_model',
    'get_content_recommendations',
    'get_copurchase_recommendations',
    'get_coquotation_recommendations',
    'get_hybrid_recommendations',
    'get_historical_recommendations',
    'test_all_models',
    'test_historical_model'
]
# Para usar la función, ejecuta: test_all_models("producto_ejemplo")
# ejemplo de uso
respuesta = test_all_models("producto_125", N=5)
print(respuesta)
