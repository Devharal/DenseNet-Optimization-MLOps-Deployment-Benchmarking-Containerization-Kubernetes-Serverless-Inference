#!/usr/bin/env python3
"""
Test client for DenseNet KNative service
Tests the deployed service with sample images
"""

import base64
import json
import requests
import time
import subprocess
from PIL import Image
import io
import numpy as np

def get_service_url():
    """Get the KNative service URL"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "ksvc", "densenet-inference", "-o", "jsonpath={.status.url}"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("❌ Could not get service URL. Is the service deployed?")
        return None

def create_test_image():
    """Create a test image (random noise that looks like an image)"""
    # Create a 224x224 RGB image with random colors
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def test_health_check(service_url):
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{service_url}/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_prediction(service_url, test_image):
    """Test the prediction endpoint"""
    print("🧠 Testing prediction...")
    
    payload = {
        "image": test_image,
        "batch_size": 1
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{service_url}/predict",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful!")
            print(f"🕐 Total latency: {(end_time - start_time) * 1000:.2f}ms")
            print(f"🔮 Model latency: {data.get('latency_ms', 'N/A')}ms")
            print(f"🏷️  Top prediction: {data['predictions'][0]['class_name']} ({data['predictions'][0]['confidence']:.4f})")
            print(f"📊 All predictions: {json.dumps(data['predictions'], indent=2)}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_root_endpoint(service_url):
    """Test the root endpoint"""
    print("🏠 Testing root endpoint...")
    try:
        response = requests.get(service_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working: {data['service']} v{data['version']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def main():
    print("🧪 Testing DenseNet KNative Service")
    print("=" * 50)
    
    # Get service URL
    service_url = get_service_url()
    if not service_url:
        return
    
    print(f"🔗 Service URL: {service_url}")
    
    # Wait a bit for service to start (cold start)
    print("⏳ Waiting for service to warm up...")
    time.sleep(5)
    
    # Create test image
    print("🖼️  Creating test image...")
    test_image = create_test_image()
    
    # Run tests
    tests = [
        ("Root endpoint", lambda: test_root_endpoint(service_url)),
        ("Health check", lambda: test_health_check(service_url)),
        ("Prediction", lambda: test_prediction(service_url, test_image))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your KNative deployment is working correctly.")
    else:
        print("🔧 Some tests failed. Check the service logs with:")
        print("kubectl logs -l serving.knative.dev/service=densenet-inference -f")

if __name__ == "__main__":
    main()