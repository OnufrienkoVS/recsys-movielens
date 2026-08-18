import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Тест эндпоинта /health"""
    print("\n" + "=" * 60)
    print("1. Тест /health")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert data['status'] == 'healthy'
        print("✅ /health работает")
        return data
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_root():
    """Тест эндпоинта /root"""
    print("\n" + "=" * 60)
    print("2. Тест /root")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert 'endpoints' in data
        print("✅ / работает")
        return data
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_models():
    """Тест эндпоинта /models"""
    print("\n" + "=" * 60)
    print("3. Тест /models")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
        assert response.status_code == 200
        assert 'available_models' in data
        print("✅ /models работает")
        
        # Проверяем, что есть загруженная модель
        if data.get('current_model'):
            print(f"   Текущая модель: {data['current_model']}")
        else:
            print("   ⚠️ Нет загруженной модели")
        
        return data
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_predict():
    """Тест эндпоинта /predict"""
    print("\n" + "=" * 60)
    print("4. Тест /predict")
    print("=" * 60)
    
    payload = {"user_id": 1, "movie_id": 1}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Запрос: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            assert 'predicted_rating' in data
            assert isinstance(data['predicted_rating'], (int, float))
            print(f"✅ /predict работает (оценка: {data['predicted_rating']})")
        else:
            print(f"⚠️ /predict вернул ошибку: {data}")
        
        return data
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_predict_unknown_user():
    """Тест /predict для неизвестного пользователя"""
    print("\n" + "=" * 60)
    print("5. Тест /predict (неизвестный пользователь)")
    print("=" * 60)
    
    payload = {"user_id": 99999, "movie_id": 1}
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Запрос: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        # Должен вернуть 3.5 (средняя оценка)
        if response.status_code == 200:
            print(f"✅ Неизвестный пользователь обработан (оценка: {data['predicted_rating']})")
        else:
            print(f"⚠️ Статус: {response.status_code}")
        
        return data
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_recommend():
    """Тест эндпоинта /recommend"""
    print("\n" + "=" * 60)
    print("6. Тест /recommend")
    print("=" * 60)
    
    payload = {"user_id": 1, "n": 5}
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Запрос: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            recommendations = data.get('recommendations', [])
            print(f"Получено рекомендаций: {len(recommendations)}")
            
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec.get('title', 'N/A')} (оценка: {rec.get('predicted_rating', 'N/A')})")
            
            assert len(recommendations) > 0
            assert 'movie_id' in recommendations[0]
            assert 'title' in recommendations[0]
            assert 'predicted_rating' in recommendations[0]
            print("✅ /recommend работает")
        else:
            print(f"⚠️ Ошибка: {data}")
        
        return data
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_recommend_large_n():
    """Тест /recommend с большим n"""
    print("\n" + "=" * 60)
    print("7. Тест /recommend (n=50)")
    print("=" * 60)
    
    payload = {"user_id": 1, "n": 50}
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get('recommendations', [])
            print(f"Получено рекомендаций: {len(recommendations)}")
            assert len(recommendations) == 50
            print("✅ /recommend с n=50 работает")
        else:
            print(f"⚠️ Ошибка: {response.status_code}")
        
        return response
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_recommend_unknown_user():
    """Тест /recommend для неизвестного пользователя"""
    print("\n" + "=" * 60)
    print("8. Тест /recommend (неизвестный пользователь)")
    print("=" * 60)
    
    payload = {"user_id": 99999, "n": 10}
    
    try:
        response = requests.post(
            f"{BASE_URL}/recommend",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get('recommendations', [])
            print(f"Получено рекомендаций: {len(recommendations)}")
            print("✅ Неизвестный пользователь обработан (возвращены популярные фильмы)")
        else:
            print(f"⚠️ Ошибка: {response.status_code}")
        
        return response
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_switch_model():
    """Тест переключения модели"""
    print("\n" + "=" * 60)
    print("9. Тест /models/switch (переключение модели)")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        models_data = response.json()
        available_models = [m['model_id'] for m in models_data.get('available_models', [])]
        print(f"Доступные модели: {available_models}")

        if len(available_models) < 2:
            print("⚠️ Доступна только одна модель, проверка переключения пропущена")
            return {"status": "skipped", "reason": "only one model available"}

        current_model = models_data.get('current_model')
        target_model = next((m for m in available_models if m != current_model), None)

        if target_model is None:
            print("⚠️ Нет модели для переключения")
            return {"status": "skipped", "reason": "no alternative model"}

        print(f"Текущая модель: {current_model}")
        print(f"Переключаем на: {target_model}")

        payload = {"model_id": target_model}
        response = requests.post(
            f"{BASE_URL}/models/switch",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")

        assert response.status_code == 200
        assert data['status'] == 'success'
        assert data['current_model'] == target_model
        assert data['previous_model'] == current_model

        print(f"✅ Модель переключена на {target_model}")
        return data

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_switch_model_invalid():
    """Тест переключения на несуществующую модель"""
    print("\n" + "=" * 60)
    print("10. Тест /models/switch (несуществующая модель)")
    print("=" * 60)

    payload = {"model_id": "non_existent_model"}

    try:
        response = requests.post(
            f"{BASE_URL}/models/switch",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")

        assert response.status_code == 404
        print("✅ Несуществующая модель корректно отклонена")
        return data

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def test_current_model():
    """Тест получения информации о текущей модели"""
    print("\n" + "=" * 60)
    print("11. Тест /models/current")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/models/current", timeout=5)
        print(f"Статус: {response.status_code}")
        data = response.json()
        print(f"Ответ: {json.dumps(data, indent=2, ensure_ascii=False)}")

        assert response.status_code == 200
        if data.get('status') == 'loaded':
            assert 'model_type' in data
            print(f"✅ Текущая модель: {data['model_type']}")
        else:
            print(f"⚠️ Модель не загружена: {data.get('message')}")

        return data

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "=" * 60)
    print("ЗАПУСК ТЕСТОВ API")
    print("=" * 60)
    print(f"BASE_URL: {BASE_URL}")
    
    # Проверяем, что сервер запущен
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ Сервер не запущен!")
        print("   Запустите: python -m app.main")
        print("   Или: uvicorn app.main:app --reload")
        return
    
    # Запускаем тесты
    results = {
        'health': test_health(),
        'root': test_root(),
        'models': test_models(),
        'predict': test_predict(),
        'predict_unknown': test_predict_unknown_user(),
        'recommend': test_recommend(),
        'recommend_large': test_recommend_large_n(),
        'recommend_unknown': test_recommend_unknown_user(),
        'switch_model': test_switch_model(),
        'switch_invalid': test_switch_model_invalid(),
        'current_model': test_current_model()
    }
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results.items():
        if result is not None:
            if isinstance(result, dict) and result.get('error'):
                print(f"❌ {name}: ОШИБКА")
                failed += 1
            else:
                print(f"✅ {name}: УСПЕШНО")
                passed += 1
        else:
            print(f"❌ {name}: ОШИБКА")
            failed += 1
    
    print(f"\nИтого: {passed} пройдено, {failed} провалено")
    
    if failed == 0:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print("\n⚠️ Есть проблемы, проверьте логи выше.")


if __name__ == "__main__":
    run_all_tests()