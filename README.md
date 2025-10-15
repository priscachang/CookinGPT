# CookinGPT - Smart Recipe Finder

An intelligent recipe recommendation system that helps users discover delicious recipes based on ingredients they have available. The system uses advanced AI embeddings and natural language processing to provide personalized recipe suggestions with detailed cooking instructions.

## 🌟 Key Features

### **Pre-loaded Recipe Database**

- **15 Curated Recipes**: Ready-to-use collection of popular recipes including pasta, stir-fries, salads, and desserts
- **No Setup Required**: System initializes automatically with embedded recipes
- **Instant Search**: Start finding recipes immediately without uploading files

### **Intelligent Ingredient Matching**

- **Semantic Search**: Uses Mistral AI embeddings for intelligent ingredient matching
- **Natural Language Processing**: Understands ingredient variations (e.g., "tomatoes" matches "tomato")
- **Hybrid Search**: Combines semantic similarity with keyword matching for optimal results
- **Smart Fallback**: Gracefully handles API failures with keyword-based search

### **Advanced LLM-Powered Parsing**

- **Natural Language Input**: Describe what you have in natural language
- **Intelligent Parsing**: AI extracts ingredients and preferences from conversational input
- **Preference Recognition**: Understands dietary restrictions, cooking time, and cuisine preferences
- **Context-Aware**: Adapts to different ways of describing ingredients

### **Modern Web Interface**

- **Beautiful UI**: Clean, responsive design with gradient backgrounds and smooth animations
- **Real-time Search**: Instant recipe recommendations with loading states
- **Match Scoring**: Visual percentage scores showing ingredient compatibility
- **Detailed Instructions**: Complete cooking steps for each recipe

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd smart-recipe-finder

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with your Mistral API key:

```bash
# Create .env file
echo "MISTRAL_API_KEY=your_mistral_api_key_here" > .env
```

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:8000` with embedded recipes ready to use!

## 💡 How to Use

### **Simple Ingredient Search**

1. Open your browser to `http://localhost:8000`
2. Enter your available ingredients (e.g., "chicken, rice, vegetables")
3. Click "Find Recipes" to get personalized recommendations
4. View match scores, ingredients needed, and cooking steps

### **Natural Language Search**

Try conversational queries like:

- "I have chicken, rice, and some vegetables in my fridge"
- "What can I make with eggs, cheese, and bread?"
- "I need something quick with ground beef and pasta"

### **Understanding Results**

- **Match Score**: Percentage showing how well your ingredients match the recipe
- **Ingredients Needed**: Complete list of ingredients required
- **Cooking Steps**: Detailed step-by-step instructions
- **Missing Ingredients**: What you might need to buy

## 🔧 Technical Architecture

### **Embedding Technology**

- **Model**: Mistral Embeddings (`mistral-embed`)
- **Vector Dimensions**: High-dimensional semantic representations
- **Similarity**: Cosine similarity for ingredient matching
- **Fallback**: Keyword-based search when embeddings unavailable

### **Search Methods**

1. **Semantic Search**: Vector similarity using Mistral embeddings
2. **Keyword Search**: Traditional text matching for fallback
3. **Hybrid Search**: Combines both methods for optimal results
4. **LLM Parsing**: Natural language understanding for complex queries

### **Data Flow**

```
User Input → LLM Parsing → Ingredient Extraction → Embedding Generation →
Vector Similarity → Recipe Ranking → Results Display
```

## 📁 Project Structure

```
├── app.py                          # FastAPI application with modern UI
├── models.py                       # Pydantic data models
├── embedded_recipes.py             # Pre-loaded recipe database
├── utils.py                        # Utility functions and embeddings
├── services/
│   ├── recipe_service.py          # Recipe management and CSV processing
│   ├── recipe_search_service.py   # Search algorithms and matching
│   └── llm_parsing_service.py     # Natural language parsing
├── static/
│   └── logo.png                   # Application logo
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🛠️ API Endpoints

### **Search Recipes (Traditional)**

```http
POST /search-recipes
Content-Type: application/json

{
  "ingredients": ["chicken", "rice", "vegetables"],
  "top_k": 5,
  "threshold": 0.6
}
```

### **Search Recipes (LLM-Powered)**

```http
POST /search-recipes-llm
Content-Type: application/json

{
  "user_input": "I have chicken, rice, and some vegetables in my fridge",
  "top_k": 5,
  "threshold": 0.6
}
```

### **Response Format**

```json
{
  "recommendations": [
    {
      "recipe_id": "recipe_001",
      "title": "Chicken Stir Fry",
      "ingredients": "chicken breast, bell peppers, broccoli, soy sauce...",
      "steps": "1. Cut chicken into strips...",
      "match_score": 0.85,
      "matched_ingredients": ["chicken", "vegetables"],
      "missing_ingredients": ["soy sauce", "bell peppers"]
    }
  ],
  "total_matches": 3,
  "processing_time": 0.45,
  "parsed_ingredients": ["chicken", "rice", "vegetables"]
}
```

## 🎯 Advanced Features

### **Embedded Recipe Collection**

The system comes with 15 pre-loaded recipes covering various cuisines:

- **Pasta Dishes**: Spaghetti Carbonara, Pasta Primavera
- **Asian Cuisine**: Chicken Stir Fry, Beef and Broccoli
- **Comfort Food**: Grilled Cheese, Chicken Noodle Soup
- **Healthy Options**: Salmon with Roasted Vegetables, Caesar Salad
- **Desserts**: Chocolate Chip Cookies
- **And more...**

### **Intelligent Ingredient Normalization**

- Handles plural/singular forms (tomato/tomatoes)
- Recognizes ingredient variations (bell pepper/pepper)
- Processes natural language descriptions
- Extracts preferences and dietary requirements

### **Robust Error Handling**

- **API Rate Limiting**: Graceful handling of API limits
- **Network Failures**: Automatic fallback to keyword search
- **Invalid Input**: Smart parsing with user feedback
- **Empty Results**: Helpful suggestions for better searches

## 🔧 Customization

### **Adding New Recipes**

1. Edit `embedded_recipes.py`
2. Add new recipe objects to `EMBEDDED_RECIPES_DATA`
3. Restart the application

### **Modifying Search Parameters**

- **Similarity Threshold**: Adjust `threshold` parameter (0.0-1.0)
- **Result Count**: Change `top_k` parameter
- **Search Method**: Toggle between semantic, keyword, or hybrid search

### **Custom Embedding Models**

- Modify `utils.py` to use different embedding providers
- Update API keys and model names
- Adjust vector dimensions as needed

## 🚨 Troubleshooting

### **Common Issues**

**No recipes found:**

- Try lowering the similarity threshold (0.3-0.5)
- Add more specific ingredients
- Check for typos in ingredient names

**API errors:**

- Verify your Mistral API key is correct
- Check your internet connection
- Ensure you have sufficient API credits

**Slow performance:**

- The first search may be slower due to embedding generation
- Subsequent searches are much faster
- Consider using fewer ingredients for faster results

### **Debug Mode**

Enable detailed logging by setting environment variables:

```bash
export DEBUG=true
python app.py
```

## 📊 Performance

- **Initialization**: ~2-3 seconds for embedded recipes
- **Search Speed**: <1 second for most queries
- **Embedding Generation**: ~0.5 seconds per query
- **Memory Usage**: ~50MB for full recipe database
- **Concurrent Users**: Supports multiple simultaneous searches

## 🔮 Future Enhancements

- **User Preferences**: Save favorite recipes and dietary restrictions
- **Recipe Scaling**: Adjust serving sizes automatically
- **Nutritional Information**: Add calorie and nutrition data
- **Image Recognition**: Upload photos to identify ingredients
- **Shopping Lists**: Generate grocery lists for missing ingredients
- **Cooking Timers**: Integrated cooking timers for each recipe

## 📄 License

This project is part of the "Business in LLM" course at Columbia University.

## 🤝 Contributing

This is an educational project. For questions or improvements, please contact the development team.

---

**Ready to discover amazing recipes?** Start the application and begin your culinary journey with Smart Recipe Finder! 🍳✨
