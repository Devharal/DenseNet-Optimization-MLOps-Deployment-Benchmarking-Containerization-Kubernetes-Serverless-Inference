# test_api.py
import requests
import base64
import json
import time
from PIL import Image
import io
import os

def create_test_image(size=(224, 224), color='red'):
    """Create a test image and return as base64 string"""
    img = Image.new('RGB', size, color=color)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_model_info_endpoint(base_url):
    """Test the model info endpoint"""
    try:
        response = requests.get(f"{base_url}/model-info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print("✅ Model info retrieved:")
            print(f"   - Model variant: {info.get('model_variant')}")
            print(f"   - Device: {info.get('device')}")
            print(f"   - Classes: {info.get('num_classes')}")
            return True
        else:
            print(f"❌ Model info failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info failed: {str(e)}")
        return False

def test_prediction_endpoint(base_url, batch_sizes=[1, 2, 4]):
    """Test the prediction endpoint with different batch sizes"""
    test_image = create_test_image()
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n🔄 Testing batch size: {batch_size}")
        
        try:
            payload = {
                "image": test_image,
                "batch_size": batch_size
            }
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/predict",
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                total_time = (end_time - start_time) * 1000
                
                print(f"   ✅ Success!")
                print(f"   - Model latency: {data['latency_ms']:.2f}ms")
                print(f"   - Total time: {total_time:.2f}ms")
                print(f"   - Model variant: {data['model_variant']}")
                print(f"   - Top prediction: {data['predictions'][0][0]['class']} ({data['predictions'][0][0]['confidence']:.3f})")
                
                results.append({
                    "batch_size": batch_size,
                    "latency_ms": data['latency_ms'],
                    "total_time_ms": total_time,
                    "model_variant": data['model_variant'],
                    "top_prediction": data['predictions'][0][0],
                    "status": "success"
                })
            else:
                print(f"   ❌ Failed with status: {response.status_code}")
                print(f"   - Response: {response.text}")
                results.append({
                    "batch_size": batch_size,
                    "status": "failed",
                    "error": response.text
                })
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                "batch_size": batch_size,
                "status": "error",
                "error": str(e)
            })
    
    return results

def run_comprehensive_test(base_url="http://localhost:30080"):
    """Run comprehensive API tests"""
    print("🚀 Starting comprehensive API tests")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint")
    health_ok = test_health_endpoint(base_url)
    
    # Test model info endpoint
    print("\n2. Testing Model Info Endpoint")
    info_ok = test_model_info_endpoint(base_url)
    
    # Test prediction endpoint
    print("\n3. Testing Prediction Endpoint")
    prediction_results = test_prediction_endpoint(base_url)
    
    # Generate summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    success_count = sum(1 for r in prediction_results if r.get('status') == 'success')
    total_tests = len(prediction_results)
    
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Model Info: {'✅ PASS' if info_ok else '❌ FAIL'}")
    print(f"Predictions: {success_count}/{total_tests} successful")
    
    if success_count > 0:
        successful_results = [r for r in prediction_results if r.get('status') == 'success']
        avg_latency = sum(r['latency_ms'] for r in successful_results) / len(successful_results)
        print(f"Average Latency: {avg_latency:.2f}ms")
    
    # Save results to file
    output_file = "api_test_results.json"
    test_summary = {
        "base_url": base_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "health_check": health_ok,
        "model_info": info_ok,
        "prediction_results": prediction_results,
        "summary": {
            "total_tests": total_tests,
            "successful_tests": success_count,
            "success_rate": success_count / total_tests if total_tests > 0 else 0
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return test_summary

def load_image_from_file(image_path):
    """Load image from file and convert to base64"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode()
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def test_with_real_image(base_url, image_path):
    """Test API with a real image file"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    print(f"\n🖼️  Testing with real image: {image_path}")
    
    image_b64 = load_image_from_file(image_path)
    if not image_b64:
        return None
    
    try:
        payload = {
            "image": image_b64,
            "batch_size": 1
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", json=payload, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            total_time = (end_time - start_time) * 1000
            
            print("✅ Prediction successful!")
            print(f"   - Model latency: {data['latency_ms']:.2f}ms")
            print(f"   - Total time: {total_time:.2f}ms")
            print("   - Top 3 predictions:")
            
            for i, pred in enumerate(data['predictions'][0][:3]):
                print(f"     {i+1}. {pred['class']}: {pred['confidence']:.3f}")
            
            return data
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    import sys
    
    # Default base URL
    base_url = "http://localhost:30080"
    
    # Override base URL if provided as argument
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"Testing API at: {base_url}")
    
    # Run comprehensive tests
    results = run_comprehensive_test(base_url)
    
    # Test with real image if provided
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
        test_with_real_image(base_url, image_path)
    
    print("\n🎉 Testing completed!")
    print(f"Access the API documentation at: {base_url}/docs")