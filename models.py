from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.6
    use_hybrid: bool = True
    intent: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    evidence_score: float
    query_type: str
    processing_time: float

class IngestionResponse(BaseModel):
    status: str
    ingested_chunks: int
    files_processed: List[str]
    total_chunks: int

# Recipe-specific models
class Recipe(BaseModel):
    id: str
    title: str
    ingredients: str
    steps: str
    embedding: Optional[List[float]] = None

class RecipeIngestionRequest(BaseModel):
    csv_file: str  # Base64 encoded CSV content or file path

class RecipeIngestionResponse(BaseModel):
    status: str
    recipes_processed: int
    total_recipes: int

class IngredientSearchRequest(BaseModel):
    ingredients: List[str]
    top_k: int = 5
    threshold: float = 0.6

class RecipeSearchRequest(BaseModel):
    user_input: str
    top_k: int = 5
    threshold: float = 0.6

class RecipeRecommendation(BaseModel):
    recipe_id: str
    title: str
    ingredients: str
    steps: str
    match_score: float
    matched_ingredients: List[str]
    missing_ingredients: List[str]

class RecipeSearchResponse(BaseModel):
    recommendations: List[RecipeRecommendation]
    total_matches: int
    processing_time: float
    parsed_ingredients: Optional[List[str]] = None
