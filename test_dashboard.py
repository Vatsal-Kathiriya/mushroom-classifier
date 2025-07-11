#!/usr/bin/env python3
"""
Test script to verify the mushroom classification dashboard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

def test_dashboard():
    """Test the dashboard route"""
    with app.test_client() as client:
        try:
            # Test the dashboard route
            response = client.get('/dashboard')
            print(f"Dashboard route status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Dashboard route is working!")
                print(f"Response length: {len(response.data)} bytes")
            else:
                print(f"❌ Dashboard route failed with status {response.status_code}")
                print(f"Response: {response.data.decode()}")
                
        except Exception as e:
            print(f"❌ Error testing dashboard: {str(e)}")

def test_api_predict():
    """Test the API predict endpoint"""
    with app.test_client() as client:
        try:
            # Test data
            test_data = {
                'cap-shape': 'x',
                'cap-color': 'n',
                'does-bruise-or-bleed': 'f',
                'gill-color': 'w',
                'habitat': 'g',
                'season': 's'
            }
            
            response = client.post('/api/predict', 
                                 json=test_data,
                                 content_type='application/json')
            
            print(f"API predict status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.get_json()
                print("✅ API predict is working!")
                print(f"Prediction: {result.get('prediction')}")
                print(f"Confidence: {result.get('confidence'):.1%}")
            else:
                print(f"❌ API predict failed with status {response.status_code}")
                print(f"Response: {response.data.decode()}")
                
        except Exception as e:
            print(f"❌ Error testing API predict: {str(e)}")

if __name__ == "__main__":
    print("🍄 Testing Mushroom Classification Dashboard")
    print("=" * 50)
    
    test_dashboard()
    print()
    test_api_predict()
    print()
    print("Test completed! Run 'python app.py' to start the server.")
