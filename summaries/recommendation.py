from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
import requests
from .models import Book, UserActivity, Category, OpenLibraryBook
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

OPEN_LIBRARY_API_URL = "https://openlibrary.org/search.json"
MAX_RESULTS = 50

# Singleton model loader to avoid reloading on every import/call
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def generate_embedding_for_text(text):
    """Generates an embedding for a single string of text."""
    if not text:
        return None
    model = get_model()
    return model.encode(text).tolist()

def update_book_embedding(book):
    """Updates the embedding for a specific book instance."""
    metadata = f"{book.title or ''} {book.author or ''} {book.content or ''}"
    categories = " ".join([c.name for c in book.categories.all()])
    full_text = f"{metadata} {categories}"
    book.embedding_vector = generate_embedding_for_text(full_text)
    book.save(update_fields=['embedding_vector'])

def fetch_open_library_books(query, max_results=10):
    """
    Fetches books from Open Library, checking local cache (OpenLibraryBook model) or cache dict first.
    """
    # Simple in-memory cache for the session (optional, can be removed if DB cache is sufficient)
    # Ideally, we should check the database first for 'similar' queries if we had a query model, 
    # but for now we will stick to request-level caching or just hitting the API efficiently.
    
    try:
        response = requests.get(f"{OPEN_LIBRARY_API_URL}?q={query}&limit={max_results}", timeout=5)
        if response.status_code == 200:
            results = response.json().get('docs', [])
            books = []
            for book_data in results:
                key = book_data.get('key', '')
                title = book_data.get('title', '')
                author = ', '.join(book_data.get('author_name', []))
                cover_id = book_data.get('cover_i')
                cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None

                # Cache this book in our DB for future detail lookups
                if key:
                     OpenLibraryBook.objects.get_or_create(
                        key=key,
                        defaults={
                            'title': title,
                            'author': author,
                            'cover_url': cover_url
                        }
                    )

                books.append({
                    "title": title,
                    "author": author,
                    "cover_url": cover_url,
                    "key": key,
                })
            return books
        return []
    except Exception as e:
        logger.error(f"Error fetching Open Library books: {e}")
        return []

def recommend_based_on_behavior(user_id, all_books, book_embeddings):
    """
    Recommends books based on user activity using pre-calculated embeddings.
    """
    activities = UserActivity.objects.filter(
        user_id=user_id, activity_type__in=['view', 'bookmark', 'detail_click'], book__isnull=False
    ).select_related('book')
    
    if not activities.exists():
        return []

    activity_weights = {'view': 1, 'bookmark': 5, 'detail_click': 3}
    
    # Create a weighted user profile vector
    user_vector = np.zeros(len(book_embeddings[0])) if book_embeddings else []
    total_weight = 0

    # Map book IDs to their embedding index
    book_id_to_index = {b.id: i for i, b in enumerate(all_books)}

    for activity in activities:
        if activity.book.id in book_id_to_index:
            weight = activity_weights.get(activity.activity_type, 1)
            idx = book_id_to_index[activity.book.id]
            embedding = book_embeddings[idx]
            if embedding:
                user_vector = np.add(user_vector, np.array(embedding) * weight)
                total_weight += weight
    
    if total_weight == 0:
        return []
        
    user_vector = user_vector / total_weight # Normalize

    # Calculate similarity with all books
    sim_scores = cosine_similarity([user_vector], book_embeddings)[0]
    
    # Pair scores with books
    scored_books = list(zip(all_books, sim_scores))
    
    # Sort by score
    scored_books.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 10 books (excluding ones user has already interacted safely if needed, but for now just top)
    return [book for book, score in scored_books[:10]]

def get_hybrid_recommendations(user):
    # Fetch all books with embeddings
    # We filter for books that actually HAVE embeddings to avoid errors
    all_books = list(Book.objects.exclude(embedding_vector__isnull=True))
    
    # If no books have embeddings, avoiding crash
    if not all_books:
        # Fallback: try to generate for some? or just return empty
        # For now, return random
        return list(Book.objects.all()[:10]), []

    book_embeddings = [b.embedding_vector for b in all_books]
    
    local_recs = []
    if user.is_authenticated:
        # Behavioral
        behavior_recs = recommend_based_on_behavior(user.id, all_books, book_embeddings)
        
        # Category based (simple filter)
        profile = getattr(user, 'userprofile', None)
        category_recs = []
        if profile:
            fav_cats = profile.preferred_categories.all()
            if fav_cats.exists():
                category_recs = list(Book.objects.filter(categories__in=fav_cats).exclude(id__in=[b.id for b in behavior_recs])[:10])

        # Combine, prioritizing behavior
        local_recs = list(dict.fromkeys(behavior_recs + category_recs))[:10]
    else:
        # Non-auth: just random or highest rated
        local_recs = list(Book.objects.filter(is_recommended=True).order_by('?')[:10])

    # If we still have few recs, fill with random
    if len(local_recs) < 5:
        remaining = list(Book.objects.exclude(id__in=[b.id for b in local_recs]).order_by('?')[:10-len(local_recs)])
        local_recs.extend(remaining)

    # External recommendations based on the top local recommendation
    open_library_recs = []
    if local_recs:
        top_book = local_recs[0]
        # Use simple title search for now to stay fast
        open_library_recs = fetch_open_library_books(top_book.title, max_results=5)

    return local_recs, open_library_recs
