INPUT_SCHEMA = {
    "query": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Organic skincare products for sensitive skin"]
    },
    "documents": {
        'datatype': 'BYTES',
        'required': True,
        'shape': [-1],
        'example': [
            "Eco-friendly kitchenware for modern homes",
            "Biodegradable cleaning supplies for eco-conscious consumers",
            "Organic cotton baby clothes for sensitive skin",
            "Natural organic skincare range for sensitive skin",
            "Tech gadgets for smart homes: 2024 edition",
            "Sustainable gardening tools and compost solutions",
            "Sensitive skin-friendly facial cleansers and toners",
            "Organic food wraps and storage solutions",
            "All-natural pet food for dogs with allergies",
            "Yoga mats made from recycled materials"
        ]
    }
}
