"""Test script to get predictions for all example prompts"""
import requests
import json

def classify_streaming(text, labels, model='Qwen 2.5 7B'):
    """Handle SSE streaming response"""
    response = requests.post('http://localhost:5000/api/zero-shot/classify', 
        json={
            'text': text,
            'candidate_labels': labels,
            'model': model,
            'use_logprobs': False
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'result' in data:
                    return data['result']
    return None

# Test all categories
categories = {
    'sentiment': {
        'labels': ['positive', 'negative', 'neutral'],
        'examples': [
            ('Major shipping alliance announces new ultra-large container vessels with 50% reduction in emissions per container. Industry leaders praise breakthrough in sustainable maritime transport.', 'positive'),
            ("Port strike enters third week as dockworkers reject latest offer. Container backlog grows to record levels, threatening supply chain collapse across major retail sectors.", 'negative'),
            ('Global container shipping rates remain stable in Q3 according to latest freight index. Trans-Pacific routes show minimal fluctuation from previous quarter.', 'neutral'),
        ]
    },
    'cargo': {
        'labels': ['standard cargo', 'refrigerated cargo', 'hazardous materials', 'oversized freight'],
        'examples': [
            ('Pharmaceutical shipment of temperature-sensitive vaccines requires uninterrupted cold chain at 2-8°C from manufacturing facility through final delivery. Advanced reefer monitoring deployed.', 'refrigerated cargo'),
            ('Container manifest shows 500 TEU of consumer electronics and textiles loaded at Shenzhen. Standard dry containers with ambient temperature storage for trans-Pacific crossing.', 'standard cargo'),
            ('Vessel carries 200 tons of lithium-ion batteries classified as UN3480 Class 9 dangerous goods. Special segregation and fire suppression protocols in effect per IMDG Code.', 'hazardous materials'),
            ('Breakbulk carrier loading 80-meter wind turbine blades onto reinforced flat racks. Route survey completed for overhead clearances through Panama Canal transit.', 'oversized freight'),
        ]
    }
}

print('=' * 80)
print('EXAMPLE PROMPT PREDICTIONS (Qwen 2.5 7B, Fast Mode)')
print('=' * 80)

for category, data in categories.items():
    print(f'\n{category.upper()}:')
    print('-' * 80)
    
    for text, expected in data['examples']:
        result = classify_streaming(text, data['labels'])
        if result:
            # Debug: print result structure
            if category == 'sentiment' and expected == 'positive':
                print(f'DEBUG result structure: {result.keys()}')
            
            predicted = result.get('top_label') or result.get('label') or result.get('predicted_label')
            match = '✓' if predicted and predicted.lower() == expected.lower() else '✗'
            confidence = result.get('top_score', 0) * 100
            print(f'{match} Expected: {expected:20s} | Predicted: {predicted} ({confidence:.1f}%)')
            print(f'  Text: {text[:60]}...')
            print()

print('=' * 80)
