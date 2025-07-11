"""
Script de prueba para verificar la implementación de SumTree y memoria priorizada.
"""

import torch
import numpy as np
import time
from replay_memory import SumTree, PrioritizedReplayMemory, ReplayMemory

def test_sumtree_basic():
    """
    Prueba básica del SumTree.
    """
    print("=== Prueba básica del SumTree ===")
    
    # Crear SumTree con capacidad 4
    tree = SumTree(4)
    
    # Agregar elementos con diferentes prioridades
    tree.add(1.0, "data1")
    tree.add(2.0, "data2")
    tree.add(3.0, "data3")
    tree.add(4.0, "data4")
    
    print(f"Total de prioridades: {tree.total()}")
    print(f"Número de entradas: {tree.n_entries}")
    
    # Probar muestreo
    print("\nMuestreo de elementos:")
    for i in range(5):
        s = np.random.uniform(0, tree.total())
        idx, priority, data = tree.get(s)
        print(f"Muestra {i}: s={s:.2f}, idx={idx}, priority={priority:.2f}, data={data}")
    
    # Probar actualización de prioridades
    print("\nActualizando prioridad del primer elemento:")
    tree.update(3, 5.0)  # Actualizar prioridad del primer elemento
    print(f"Nuevo total: {tree.total()}")
    
    # Muestrear después de la actualización
    s = np.random.uniform(0, tree.total())
    idx, priority, data = tree.get(s)
    print(f"Muestra después de actualización: s={s:.2f}, idx={idx}, priority={priority:.2f}, data={data}")

def test_prioritized_memory():
    """
    Prueba de la memoria priorizada con SumTree.
    """
    print("\n=== Prueba de memoria priorizada ===")
    
    # Crear memoria priorizada
    memory = PrioritizedReplayMemory(capacity=1000, alpha=0.6, beta=0.4)
    
    # Agregar transiciones de ejemplo
    print("Agregando transiciones...")
    for i in range(100):
        state = torch.randn(4, 84, 84)
        action = np.random.randint(0, 4)
        reward = np.random.uniform(-1, 1)
        done = np.random.choice([True, False])
        next_state = torch.randn(4, 84, 84)
        
        memory.add(state, action, reward, done, next_state)
    
    print(f"Transiciones en memoria: {len(memory)}")
    print(f"Total de prioridades: {memory.sum_tree.total():.2f}")
    
    # Probar muestreo
    print("\nMuestreando batch...")
    batch, indices, weights = memory.sample(batch_size=32)
    
    print(f"Tamaño del batch: {len(batch)}")
    print(f"Índices: {indices[:5]}...")  # Mostrar primeros 5 índices
    print(f"Pesos: {weights[:5]}...")    # Mostrar primeros 5 pesos
    
    # Probar actualización de prioridades
    print("\nActualizando prioridades...")
    td_errors = np.random.uniform(0, 10, size=32)
    memory.update_priorities(indices, td_errors)
    
    print(f"Nuevo total de prioridades: {memory.sum_tree.total():.2f}")

def test_performance_comparison():
    """
    Comparación de rendimiento entre implementaciones.
    """
    print("\n=== Comparación de rendimiento ===")
    
    capacity = 10000
    batch_size = 32
    n_samples = 1000
    
    # Crear memorias
    regular_memory = ReplayMemory(capacity)
    prioritized_memory = PrioritizedReplayMemory(capacity)
    
    # Llenar memorias con datos de ejemplo
    print("Llenando memorias...")
    for i in range(capacity):
        state = torch.randn(4, 84, 84)
        action = np.random.randint(0, 4)
        reward = np.random.uniform(-1, 1)
        done = np.random.choice([True, False])
        next_state = torch.randn(4, 84, 84)
        
        regular_memory.add(state, action, reward, done, next_state)
        prioritized_memory.add(state, action, reward, done, next_state)
    
    # Medir tiempo de muestreo para memoria regular
    print("Probando memoria regular...")
    start_time = time.time()
    for _ in range(n_samples):
        regular_memory.sample(batch_size)
    regular_time = time.time() - start_time
    
    # Medir tiempo de muestreo para memoria priorizada
    print("Probando memoria priorizada...")
    start_time = time.time()
    for _ in range(n_samples):
        prioritized_memory.sample(batch_size)
    prioritized_time = time.time() - start_time
    
    print(f"\nResultados:")
    print(f"Memoria regular: {regular_time:.4f} segundos para {n_samples} muestras")
    print(f"Memoria priorizada: {prioritized_time:.4f} segundos para {n_samples} muestras")
    print(f"Ratio: {prioritized_time/regular_time:.2f}x más lenta")
    
    # Verificar que la memoria priorizada es más lenta pero manejable
    if prioritized_time < regular_time * 10:  # No más de 10x más lenta
        print("✅ Rendimiento aceptable para memoria priorizada")
    else:
        print("⚠️ Memoria priorizada puede ser muy lenta")

def test_memory_consistency():
    """
    Prueba de consistencia de la memoria.
    """
    print("\n=== Prueba de consistencia ===")
    
    memory = PrioritizedReplayMemory(capacity=100, alpha=0.6, beta=0.4)
    
    # Agregar transiciones con prioridades conocidas
    for i in range(10):
        state = torch.tensor([i], dtype=torch.float32)
        memory.add(state, i, i, False, state)
    
    print(f"Transiciones agregadas: {len(memory)}")
    print(f"Total de prioridades: {memory.sum_tree.total():.2f}")
    
    # Verificar que el muestreo funciona
    try:
        batch, indices, weights = memory.sample(batch_size=5)
        print(f"✅ Muestreo exitoso: {len(batch)} elementos")
        
        # Verificar que los pesos están normalizados
        if weights.max() == 1.0:
            print("✅ Pesos normalizados correctamente")
        else:
            print("⚠️ Pesos no normalizados correctamente")
            
    except Exception as e:
        print(f"❌ Error en muestreo: {e}")
    
    # Verificar actualización de prioridades
    try:
        td_errors = np.ones(5) * 2.0  # Error TD constante
        memory.update_priorities(indices, td_errors)
        print("✅ Actualización de prioridades exitosa")
    except Exception as e:
        print(f"❌ Error en actualización: {e}")

def test_edge_cases():
    """
    Prueba de casos extremos.
    """
    print("\n=== Prueba de casos extremos ===")
    
    # Memoria muy pequeña
    memory = PrioritizedReplayMemory(capacity=1)
    state = torch.tensor([1.0])
    memory.add(state, 0, 1.0, False, state)
    
    try:
        batch, indices, weights = memory.sample(batch_size=1)
        print("✅ Muestreo con capacidad 1 funciona")
    except Exception as e:
        print(f"❌ Error con capacidad 1: {e}")
    
    # Batch size mayor que memoria
    try:
        batch, indices, weights = memory.sample(batch_size=2)
        print("❌ Debería haber fallado con batch_size > capacidad")
    except ValueError:
        print("✅ Correctamente rechaza batch_size > capacidad")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
    
    # Memoria vacía
    memory.clear()
    try:
        batch, indices, weights = memory.sample(batch_size=1)
        print("❌ Debería haber fallado con memoria vacía")
    except ValueError:
        print("✅ Correctamente rechaza muestreo de memoria vacía")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    print("Iniciando pruebas de SumTree y memoria priorizada...")
    print("=" * 60)
    
    try:
        test_sumtree_basic()
        test_prioritized_memory()
        test_performance_comparison()
        test_memory_consistency()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ Todas las pruebas completadas exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc() 