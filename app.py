import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from models import RecipeIngestionResponse, IngredientSearchRequest, RecipeSearchRequest, RecipeSearchResponse
from services.recipe_service import process_csv_recipes, generate_ingredient_embeddings, save_recipes_to_kb
from services.recipe_search_service import hybrid_recipe_search, search_recipes_with_llm_parsing
from utils import get_embedding
from mistralai import Mistral

app = FastAPI(title="Smart Recipe Finder", description="AI-powered recipe recommendation system")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load environment variables
load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable is required")

llm_client = Mistral(api_key=MISTRAL_API_KEY)

# Recipe endpoints
@app.post("/ingest-recipes", response_model=RecipeIngestionResponse)
async def ingest_recipes_csv(files: list[UploadFile] = File(...)):
    """Ingest recipe CSV files and create embeddings for ingredients."""
    import time
    start_time = time.time()
    
    processed_files = []
    total_recipes = 0
    
    for file in files:
        try:
            if not file.filename.endswith('.csv'):
                print(f"Skipping {file.filename}: Not a CSV file")
                continue
            
            # Save uploaded file temporarily
            temp_file_path = f"temp_{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process CSV file
            recipes = process_csv_recipes(temp_file_path)
            print(f"Processed {len(recipes)} recipes from {file.filename}")
            
            # Generate embeddings for ingredients
            recipes_with_embeddings = generate_ingredient_embeddings(recipes)
            
            # Save to recipe knowledge base
            save_recipes_to_kb(recipes_with_embeddings)
            
            processed_files.append(file.filename)
            total_recipes += len(recipes_with_embeddings)
            
            # Clean up temp file
            os.remove(temp_file_path)
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue
    
    if not processed_files:
        raise HTTPException(status_code=400, detail="No valid CSV files processed")
    
    return RecipeIngestionResponse(
        status="success",
        recipes_processed=total_recipes,
        total_recipes=total_recipes
    )

@app.post("/search-recipes", response_model=RecipeSearchResponse)
async def search_recipes_by_ingredients(request: IngredientSearchRequest):
    """Search for recipes based on available ingredients."""
    import time
    start_time = time.time()
    
    try:
        # Find recipes using hybrid search
        recommendations = hybrid_recipe_search(
            user_ingredients=request.ingredients,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return RecipeSearchResponse(
            recommendations=recommendations,
            total_matches=len(recommendations),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        print(f"Error in recipe search: {e}")
        raise HTTPException(status_code=500, detail=f"Recipe search failed: {str(e)}")

@app.post("/search-recipes-llm", response_model=RecipeSearchResponse)
async def search_recipes_with_llm(request: RecipeSearchRequest):
    """Search for recipes using LLM parsing to extract ingredients from natural language input."""
    import time
    from services.llm_parsing_service import llm_parsing_service
    
    start_time = time.time()
    
    try:
        # Parse user input to get ingredients
        parsed_ingredients, preferences = llm_parsing_service.parse_user_input(request.user_input)
        
        # Find recipes using LLM parsing and hybrid search
        recommendations = search_recipes_with_llm_parsing(
            user_input=request.user_input,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return RecipeSearchResponse(
            recommendations=recommendations,
            total_matches=len(recommendations),
            processing_time=time.time() - start_time,
            parsed_ingredients=parsed_ingredients
        )
        
    except Exception as e:
        print(f"Error in LLM recipe search: {e}")
        raise HTTPException(status_code=500, detail=f"LLM recipe search failed: {str(e)}")

# UI endpoint
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve a beautiful and user-friendly chat UI with Markdown support."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Recipe Finder</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="icon" type="image/png" href="/static/favicon.png">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body { 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                min-height: 100vh;
                padding: 0;
                margin: 0;
                color: #1e293b;
                line-height: 1.7;
                font-size: 16px;
                font-weight: 400;
            }
            
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: #ffffff;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                border-radius: 0;
                overflow: hidden;
            }
            
            .header { 
                background: linear-gradient(135deg, #ffcc5e 0%, #ff9f6b 100%);
                color: white; 
                padding: 3rem 2rem;
                border-bottom: none;
                position: relative;
                overflow: hidden;
            }
            
            .header-content {
                position: relative;
                z-index: 2;
                max-width: 1000px;
                margin: 0 auto;
            }
            
            .logo-section {
                display: flex;
                align-items: center;
                gap: 1.5rem;
                margin-bottom: 1rem;
            }
            
            .header-logo {
                height: 60px;
                width: auto;
                max-width: 200px;
                object-fit: contain;
                filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
                transition: all 0.3s ease;
            }
            
            .header-logo:hover {
                transform: scale(1.05);
            }
            
            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(255,255,255,0.1) 100%);
                animation: shimmer 3s ease-in-out infinite;
            }
            
            @keyframes shimmer {
                0%, 100% { transform: translateX(-100%); }
                50% { transform: translateX(100%); }
            }
            
            .header h1 {
                font-size: 2.75rem;
                font-weight: 800;
                margin: 0;
                color: #ffffff;
                letter-spacing: -0.025em;
                line-height: 1.2;
            }
            
            .header p {
                font-size: 1.25rem;
                color: rgba(255, 255, 255, 0.85);
                margin: 0;
                font-weight: 400;
                line-height: 1.6;
            }
            
            
            .main-content {
                display: grid;
                grid-template-columns: 400px 1fr;
                min-height: calc(100vh - 200px);
                gap: 0;
            }
            
            .sidebar {
                background: #f8fafc;
                border-right: 1px solid #e2e8f0;
                padding: 2.5rem;
                overflow-y: auto;
            }
            
            .chat-area {
                flex: 1;
                display: flex;
                flex-direction: column;
            }
            
            .upload-section {
                background: #ffffff;
                border-radius: 16px;
                padding: 2rem; 
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                border: 1px solid #e2e8f0;
                transition: all 0.3s ease;
            }
            
            .upload-section:hover {
                border-color: #FF6B2C;
                box-shadow: 0 10px 25px -3px rgba(255, 107, 44, 0.1), 0 4px 6px -2px rgba(255, 107, 44, 0.05);
                transform: translateY(-2px);
            }
            
            .upload-section h3 {
                color: #1e293b;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                font-size: 1.25rem;
                font-weight: 700;
                letter-spacing: -0.025em;
            }
            
            .upload-box {
                width: 100%;
                padding: 2.5rem;
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                background: #f8fafc;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
                margin: 1.5rem 0;
            }
            
            .upload-box:hover {
                border-color: #FF6B2C;
                background: linear-gradient(135deg, #fff7ed, #fed7aa);
                transform: scale(1.01);
            }
            
            .upload-icon {
                font-size: 2rem;
                color: #64748b;
                margin-bottom: 0.75rem;
            }
            
            .upload-text {
                font-size: 1rem;
                color: #64748b;
                margin-bottom: 1.5rem;
                font-weight: 500;
                line-height: 1.6;
            }
            
            .browse-btn {
                display: inline-block;
                padding: 0.5rem 1rem;
                background: linear-gradient(135deg, #FF6B2C, #FFB68A);
                color: white;
                border-radius: 8px;
                cursor: pointer;
                text-decoration: none;
                font-size: 0.875rem;
                font-weight: 500;
                transition: all 0.3s ease;
                border: none;
                box-shadow: 0 2px 8px rgba(255, 107, 44, 0.3);
                position: relative;
                overflow: hidden;
            }
            
            .browse-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .browse-btn:hover::before {
                left: 100%;
            }
            
            .browse-btn:hover {
                background: linear-gradient(135deg, #e55a1f, #ff9f6b);
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(255, 107, 44, 0.4);
            }
            
            .browse-btn:focus {
                outline: 2px solid #FF6B2C;
                outline-offset: 2px;
            }
            
            .upload-btn {
                width: 100%;
                padding: 0.75rem;
                background: linear-gradient(135deg, #FF6B2C, #FFB68A);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                font-size: 0.875rem;
                transition: all 0.3s ease;
                margin-top: 1rem;
                box-shadow: 0 2px 8px rgba(255, 107, 44, 0.3);
                position: relative;
                overflow: hidden;
            }
            
            .upload-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .upload-btn:hover::before {
                left: 100%;
            }
            
            .upload-btn:hover {
                background: linear-gradient(135deg, #e55a1f, #ff9f6b);
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(255, 107, 44, 0.4);
            }
            
            .upload-btn:focus {
                outline: 2px solid #FF6B2C;
                outline-offset: 2px;
            }
            
            .upload-options {
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }
            
            .upload-options .upload-btn {
                flex: 1;
                font-size: 0.9em;
                padding: 10px;
            }
            
            .ingredient-search-section {
                background: #2c2c2c;
                border-radius: 15px;
                padding: 20px; 
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            .ingredient-search-section h3 {
                color: #ffffff;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .ingredient-input {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .ingredient-input label {
                color: #ffffff; 
                font-weight: 600;
                font-size: 0.9em;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .ingredient-input input {
                padding: 12px;
                border: 2px solid #404040;
                border-radius: 8px;
                background: #1e1e1e;
                color: #ffffff;
                font-size: 0.9em;
                transition: all 0.3s ease;
            }
            
            .ingredient-input input:focus {
                border-color: #ff6b35;
                box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.2);
                background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
            }
            
            .search-recipes-btn {
                padding: 12px;
                background: linear-gradient(135deg, #e67e22, #f39c12);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }
            
            .search-recipes-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(230, 126, 34, 0.4);
            }
            
            .main-recipe-area {
                background: #ffffff;
                display: flex; 
                flex-direction: column;
                overflow: hidden;
            }
            
            .ingredient-search-main {
                background: #f8fafc;
                padding: 4rem 2rem;
                border-bottom: 1px solid #e2e8f0;
            }
            
            .main-search-box h2 {
                color: #1e293b;
                margin-bottom: 2.5rem;
                text-align: center;
                font-size: 2.25rem;
                font-weight: 800;
                letter-spacing: -0.025em;
                line-height: 1.2;
            }
            
            .ingredient-input-main {
                display: flex;
                gap: 1.5rem;
                align-items: center;
                max-width: 1000px;
                margin: 0 auto;
            }
            
            .ingredient-input-main input {
                flex: 1;
                padding: 1.5rem 2rem;
                border: 2px solid #d1d5db;
                border-radius: 16px;
                font-size: 1.25rem;
                outline: none;
                transition: all 0.3s ease;
                background: #ffffff;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                font-weight: 500;
            }
            
            .ingredient-input-main input:focus {
                border-color: #FF6B2C;
                box-shadow: 0 0 0 4px rgba(255, 107, 44, 0.2), 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                background: #ffffff;
                transform: translateY(-2px);
            }
            
            .main-search-btn {
                padding: 1.5rem 2.5rem;
                background: linear-gradient(135deg, #FF6B2C, #FFB68A);
                color: white;
                border: none;
                border-radius: 16px;
                cursor: pointer;
                font-weight: 700;
                font-size: 1.25rem;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                box-shadow: 0 10px 25px -3px rgba(255, 107, 44, 0.3), 0 4px 6px -2px rgba(255, 107, 44, 0.1);
                position: relative;
                overflow: hidden;
                min-width: 220px;
                letter-spacing: -0.025em;
            }
            
            .main-search-btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .main-search-btn:hover::before {
                left: 100%;
            }
            
            .main-search-btn:hover {
                background: linear-gradient(135deg, #e55a1f, #ff9f6b);
                transform: translateY(-4px);
                box-shadow: 0 20px 40px -4px rgba(255, 107, 44, 0.4), 0 8px 16px -4px rgba(255, 107, 44, 0.2);
            }
            
            .main-search-btn:focus {
                outline: 3px solid #FF6B2C;
                outline-offset: 3px;
            }
            
            .results-container {
                flex: 1;
                overflow-y: auto; 
                padding: 3rem;
                background: #ffffff;
            }
            
            .welcome-message {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
            }
            
            .welcome-card {
                background: #ffffff;
                padding: 4rem;
                border-radius: 24px;
                text-align: center;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                max-width: 800px;
                border: 1px solid #e2e8f0;
            }
            
            .welcome-logo-section {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            
            .welcome-logo {
                height: 80px;
                width: auto;
                max-width: 200px;
                object-fit: contain;
                filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.2));
                transition: all 0.3s ease;
            }
            
            .welcome-logo:hover {
                transform: scale(1.05);
            }
            
            .welcome-card i {
                font-size: 4rem;
                background: linear-gradient(135deg, #FF6B2C, #FFB68A);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                filter: drop-shadow(0 2px 4px rgba(255, 107, 44, 0.3));
            }
            
            .welcome-card h3 {
                color: #1e293b;
                margin-bottom: 1.5rem;
                font-size: 2rem;
                font-weight: 800;
                letter-spacing: -0.025em;
                line-height: 1.2;
            }
            
            .welcome-card p {
                color: #64748b;
                margin-bottom: 2.5rem;
                line-height: 1.7;
                font-size: 1.125rem;
                font-weight: 400;
            }
            
            .steps {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }
            
            .step {
                display: flex; 
                align-items: center;
                gap: 1.5rem;
                padding: 2rem;
                background: #f8fafc;
                border-radius: 16px;
                border-left: 4px solid #FF6B2C;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            
            .step:hover {
                background: #ffffff;
                transform: translateX(8px);
                box-shadow: 0 10px 25px -3px rgba(255, 107, 44, 0.1), 0 4px 6px -2px rgba(255, 107, 44, 0.05);
            }
            
            .step i {
                color: #FF6B2C;
                margin-right: 0.75rem;
                font-size: 1.25rem;
            }
            
            .step-number {
                background: linear-gradient(135deg, #FF6B2C, #FFB68A);
                color: white;
                width: 3rem;
                height: 3rem;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 1.125rem;
                box-shadow: 0 4px 12px rgba(255, 107, 44, 0.3);
                transition: all 0.3s ease;
            }
            
            .step-number:hover {
                transform: scale(1.15);
                box-shadow: 0 8px 20px rgba(255, 107, 44, 0.4);
            }
            
            .search-results-header {
                text-align: center;
                margin-bottom: 2rem;
                padding: 1.5rem;
                background: #f8fafc;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
            
            .search-results-header h2 {
                color: #1e293b;
                margin-bottom: 0.5rem;
                font-size: 1.25rem;
                font-weight: 600;
            }
            
            .recipe-card {
                background: linear-gradient(135deg, #ffffff, #FAFAF8);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 8px rgba(255, 107, 44, 0.1);
                border-left: 4px solid #FF6B2C;
                transition: all 0.3s ease;
                border: 1px solid #e2e8f0;
                position: relative;
                overflow: hidden;
            }
            
            .recipe-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(135deg, rgba(255, 107, 44, 0.05), rgba(255, 182, 138, 0.05));
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .recipe-card:hover::before {
                opacity: 1;
            }
            
            .recipe-card:hover {
                box-shadow: 0 8px 25px rgba(255, 107, 44, 0.2);
                transform: translateY(-3px);
            }
            
            .recipe-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #e2e8f0;
            }
            
            .recipe-header h3 {
                color: #222222;
                margin: 0;
                font-size: 1.25rem;
                font-weight: 600;
            }
            
            .match-score {
                background: #10b981;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 6px;
                font-weight: 500;
                font-size: 0.875rem;
            }
            
            .recipe-ingredients, .cooking-steps {
                margin-bottom: 1.5rem;
            }
            
            .recipe-ingredients h4, .cooking-steps h4 {
                color: #222222;
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 1rem;
                font-weight: 600;
            }
            
            .recipe-ingredients p {
                color: #555555;
                line-height: 1.6;
                background: #F4F4F4;
                padding: 1rem;
                border-radius: 6px;
                border-left: 3px solid #FF6B2C;
                font-size: 0.875rem;
            }
            
            .steps-content {
                background: #F4F4F4;
                padding: 1rem;
                border-radius: 6px;
                border-left: 3px solid #FF6B2C;
                white-space: pre-line;
                line-height: 1.6;
                color: #555555;
                font-family: inherit;
                font-size: 0.875rem;
            }
            
            .loading-container {
                text-align: center;
                padding: 3rem;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                border: 1px solid #e2e8f0;
            }
            
            .loading-container h3 {
                color: #1e293b; 
                margin: 1.5rem 0 0.75rem 0;
                font-size: 1.125rem;
                font-weight: 600;
            }
            
            .loading-container p {
                color: #64748b;
                font-size: 0.875rem;
            }
            
            .no-results, .error-message {
                text-align: center;
                padding: 3rem;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                border: 1px solid #e2e8f0;
            }
            
            .no-results i, .error-message i {
                font-size: 2rem;
                color: #64748b;
                margin-bottom: 1rem;
            }
            
            .error-message i {
                color: #ef4444;
            }
            
            .no-results h3, .error-message h3 {
                color: #1e293b;
                margin-bottom: 0.75rem;
                font-size: 1.125rem;
                font-weight: 600;
            }
            
            .no-results p, .error-message p {
                color: #64748b;
                font-size: 0.875rem;
                line-height: 1.6;
            }
            
            .success-message {
                text-align: center;
                padding: 3rem;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                border: 1px solid #e2e8f0;
            }
            
            .success-message i {
                font-size: 2rem;
                color: #10b981;
                margin-bottom: 1rem;
            }
            
            .success-message h3 {
                color: #1e293b; 
                margin-bottom: 0.75rem;
                font-size: 1.125rem;
                font-weight: 600;
            }
            
            .success-message p {
                color: #64748b;
                font-size: 0.875rem;
                line-height: 1.6;
            }
            
            .status-message {
                padding: 0.75rem;
                border-radius: 6px;
                margin: 0.75rem 0;
                font-size: 0.875rem;
            }
            
            .status-success {
                background: #dcfce7;
                color: #166534;
                border: 1px solid #bbf7d0;
            }
            
            .status-error {
                background: #fef2f2;
                color: #dc2626;
                border: 1px solid #fecaca;
            }
            
            .status-info {
                background: #eff6ff;
                color: #1e40af;
                border: 1px solid #bfdbfe;
            }
            
            
            .loading {
                display: inline-block;
                width: 1rem;
                height: 1rem;
                border: 2px solid #e2e8f0;
                border-top: 2px solid #FF6B2C;
                border-radius: 50%;
                animation: spin 1s linear infinite, pulse 2s ease-in-out infinite;
                margin-right: 0.5rem;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @media (max-width: 768px) {
                .main-content {
                    flex-direction: column;
                }
                
                .sidebar {
                    width: 100%;
                    order: 2;
                }
                
                .chat-area {
                    order: 1;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .message {
                    max-width: 95%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-content">
                    <div class="logo-section">
                        <img src="/static/logo.png" alt="Logo" class="header-logo" onerror="this.style.display='none'">
                        <h1>Smart Recipe Finder</h1>
                    </div>
                    <p>Enter ingredients you have and discover delicious recipes you can make right now!</p>
                </div>
            </div>
            
            <div class="main-content">
                <div class="sidebar">
                    <div class="upload-section">
                        <h3><i class="fas fa-upload"></i> Upload Recipe Database</h3>
                        <div class="upload-box" id="uploadArea">
                            <div class="upload-icon">📄</div>
                            <div class="upload-text">Drag & Drop your CSV recipe file here</div>
                            <label class="browse-btn" for="fileInput">Browse CSV File</label>
                            <input type="file" id="fileInput" accept=".csv" style="display: none;">
                        </div>
                        <button class="upload-btn" onclick="uploadRecipes()" id="uploadBtn" disabled>
                            <i class="fas fa-utensils"></i> Upload Recipes
                        </button>
                        <div id="uploadStatus"></div>
                    </div>
                        </div>
                        
                <div class="main-recipe-area">
                    <div class="ingredient-search-main">
                        <div class="main-search-box">
                            <h2><i class="fas fa-search"></i> What ingredients do you have?</h2>
                            <div class="ingredient-input-main">
                                <input type="text" id="mainIngredientInput" placeholder="Enter ingredients separated by commas (e.g., chicken, rice, tomatoes, onions)" onkeypress="handleIngredientKeyPress(event)">
                                <button class="main-search-btn" onclick="searchRecipes()">
                                    <i class="fas fa-utensils"></i> Find Recipes
                                </button>
                            </div>
                        </div>
                        </div>
                        
                    <div class="results-container" id="resultsContainer">
                        <div class="welcome-message">
                            <div class="welcome-card">
                                <div class="welcome-logo-section">
                                    <img src="/static/logo.png" alt="Logo" class="welcome-logo" onerror="this.style.display='none'">
                                </div>
                                <h3>Welcome to Smart Recipe Finder!</h3>
                                <p>Upload your recipe database and discover amazing recipes with your ingredients.</p>
                                <div class="steps">
                                    <div class="step">
                                        <span class="step-number">1</span>
                                        <span><i class="fas fa-upload"></i> Upload CSV</span>
                        </div>
                                    <div class="step">
                                        <span class="step-number">2</span>
                                        <span><i class="fas fa-search"></i> Enter ingredients</span>
                    </div>
                                    <div class="step">
                                        <span class="step-number">3</span>
                                        <span><i class="fas fa-magic"></i> Get recipes!</span>
                </div>
                        </div>
                    </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Configure marked for better rendering
            marked.setOptions({
                breaks: true,
                gfm: true
            });
            
            // Make functions available globally
            window.searchRecipes = function() {
                console.log('Search function called');
                const ingredientInput = document.getElementById('mainIngredientInput');
                const ingredients = ingredientInput.value.trim();
                
                console.log('Ingredients:', ingredients);
                
                if (!ingredients) {
                    alert('Please enter some ingredients first!');
                    return;
                }
                
                const ingredientList = ingredients.split(',').map(ing => ing.trim()).filter(ing => ing);
                
                if (ingredientList.length === 0) {
                    alert('Please enter valid ingredients!');
                    return;
                }
                
                // Show loading in results container
                const resultsContainer = document.getElementById('resultsContainer');
                resultsContainer.innerHTML = `
                    <div class="loading-container">
                        <div class="loading"></div>
                        <h3>Searching for recipes...</h3>
                        <p>Finding the best recipes for your ingredients: ${ingredientList.join(', ')}</p>
                    </div>
                `;
                
                console.log('Loading state set, making API call...');
                
                // Make the API call using LLM parsing
                fetch('/search-recipes-llm', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_input: ingredients,
                        top_k: 10,
                        threshold: 0.6
                    })
                })
                .then(response => response.json())
                .then(result => {
                    console.log('Response result:', result);
                    
                    if (result.recommendations && result.recommendations.length > 0) {
                        // Use parsed ingredients from LLM output1 if available, otherwise fall back to original input
                        const displayIngredients = result.parsed_ingredients && result.parsed_ingredients.length > 0 
                            ? result.parsed_ingredients 
                            : ingredientList;
                        
                        let recipesHtml = `
                            <div class="search-results-header">
                                <h2><i class="fas fa-utensils"></i> Recipe Recommendations</h2>
                                <p>Found ${result.recommendations.length} recipes for your ingredients: <strong>${displayIngredients.join(', ')}</strong></p>
                            </div>
                        `;
                        
                        result.recommendations.forEach((recipe, index) => {
                            const matchPercentage = (recipe.match_score * 100).toFixed(1);
                            
                            // Just use the steps as is
                            const formattedSteps = recipe.steps;
                            
                            recipesHtml += `
                                <div class="recipe-card">
                                    <div class="recipe-header">
                                        <h3>${recipe.title}</h3>
                                        <div class="match-score">${matchPercentage}% match</div>
                                    </div>
                                    <div class="recipe-ingredients">
                                        <h4><i class="fas fa-list"></i> Ingredients Needed:</h4>
                                        <p>${recipe.ingredients}</p>
                                    </div>
                                    <div class="cooking-steps">
                                        <h4><i class="fas fa-utensils"></i> Cooking Steps:</h4>
                                        <div class="steps-content">${formattedSteps}</div>
                                    </div>
                                </div>
                            `;
                        });
                        
                        resultsContainer.innerHTML = recipesHtml;
                    } else {
                        resultsContainer.innerHTML = `
                            <div class="no-results">
                                <i class="fas fa-search"></i>
                                <h3>No recipes found</h3>
                                <p>No recipes match your ingredients. Try adding more ingredients or lowering the similarity threshold.</p>
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    resultsContainer.innerHTML = `
                        <div class="error-message">
                            <i class="fas fa-exclamation-triangle"></i>
                            <h3>Error</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                });
            }
            
            // Handle Enter key in ingredient input - made global
            window.handleIngredientKeyPress = function(event) {
                console.log('Key pressed:', event.key);
                if (event.key === 'Enter') {
                    console.log('Enter key detected, calling searchRecipes');
                    searchRecipes();
                }
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                const uploadArea = document.getElementById('uploadArea');
                const fileInput = document.getElementById('fileInput');
                const uploadBtn = document.getElementById('uploadBtn');
                const uploadStatus = document.getElementById('uploadStatus');
                
                let selectedFile = null;
                
                // Click to select file
                uploadArea.addEventListener('click', () => {
                    fileInput.click();
                });
                
                // File input change
                fileInput.addEventListener('change', (e) => {
                    if (e.target.files.length > 0) {
                        selectedFile = e.target.files[0];
                        if (selectedFile.name.endsWith('.csv')) {
                            uploadBtn.disabled = false;
                            uploadStatus.innerHTML = `<div class="status-message status-info"><i class="fas fa-check-circle"></i> Selected: ${selectedFile.name}</div>`;
                        } else {
                            uploadBtn.disabled = true;
                            uploadStatus.innerHTML = '<div class="status-message status-error"><i class="fas fa-exclamation-triangle"></i> Please select a CSV file</div>';
                        }
                    }
                });
                
                // Drag and drop functionality
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea.classList.add('dragover');
                });
                
                uploadArea.addEventListener('dragleave', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                });
                
                uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                    if (e.dataTransfer.files.length > 0) {
                        selectedFile = e.dataTransfer.files[0];
                        if (selectedFile.name.endsWith('.csv')) {
                        uploadBtn.disabled = false;
                            uploadStatus.innerHTML = `<div class="status-message status-info"><i class="fas fa-check-circle"></i> Selected: ${selectedFile.name}</div>`;
                    } else {
                        uploadBtn.disabled = true;
                            uploadStatus.innerHTML = '<div class="status-message status-error"><i class="fas fa-exclamation-triangle"></i> Please select a CSV file</div>';
                        }
                    }
                });
                
                // Upload recipes function
                window.uploadRecipes = async function() {
                    if (!selectedFile) {
                        uploadStatus.innerHTML = '<div class="status-message status-error"><i class="fas fa-exclamation-triangle"></i> Please select a CSV file first</div>';
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('files', selectedFile);
                    
                    uploadStatus.innerHTML = '<div class="status-message status-info"><div class="loading"></div> Uploading and processing recipes...</div>';
                    
                    try {
                        const response = await fetch('/ingest-recipes', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (response.ok) {
                            uploadStatus.innerHTML = `<div class="status-message status-success"><i class="fas fa-check-circle"></i> Successfully processed ${result.recipes_processed} recipes!</div>`;
                            
                            // Show success message in results area
                            const resultsContainer = document.getElementById('resultsContainer');
                            resultsContainer.innerHTML = `
                                <div class="success-message">
                                    <i class="fas fa-check-circle"></i>
                                    <h3>Recipes Uploaded Successfully!</h3>
                                    <p>${result.recipes_processed} recipes are now available. You can start searching for recipes by entering your ingredients above.</p>
                                </div>
                            `;
                            
                            // Clear selected file
                            selectedFile = null;
                            uploadBtn.disabled = true;
                            fileInput.value = '';
                        } else {
                            uploadStatus.innerHTML = `<div class="status-message status-error"><i class="fas fa-times-circle"></i> Error: ${result.detail || 'Upload failed'}</div>`;
                        }
                    } catch (error) {
                        uploadStatus.innerHTML = `<div class="status-message status-error"><i class="fas fa-times-circle"></i> Upload failed: ${error.message}</div>`;
                    }
                };
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
