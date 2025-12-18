from django.shortcuts import render,redirect,get_object_or_404,resolve_url
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login,authenticate
from django.views.generic import TemplateView
from django.shortcuts import redirect
from django.contrib.auth.views import LoginView
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .models import *
import requests
import json
from .utils import *
from django.db.models import Q,Avg
from django.core.cache import cache
import spacy
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import api_view,permission_classes
from .serializers import BookSerializer,UserSerializer,UserProfileSerializer,UserActivitySerializer
from rest_framework import viewsets,status,generics
from django.http import JsonResponse,Http404,HttpResponse, HttpResponseBadRequest
import PyPDF2
from .forms import BookUploadForm,UserEditForm,CustomUserCreationForm,UserProfileForm
from django.views.decorators.csrf import csrf_protect,csrf_exempt,ensure_csrf_cookie
from django.contrib.auth.decorators import login_required, user_passes_test
import fitz
from .recommendation import *
from textblob import TextBlob
from django.views.decorators.http import require_POST
import logging
from django.middleware.csrf import get_token
from django.utils.decorators import method_decorator
from django.contrib.auth.models import User
from django.db.models import Count
from django.views.generic import ListView
from django.core.files.storage import default_storage
from django.contrib import messages
import os
import concurrent.futures
from pathlib import Path
from django.urls import reverse
from django.http import FileResponse
from django.views.decorators.clickjacking import xframe_options_exempt
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,pipeline,AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI
from langchain.chat_models import ChatOpenAI
from django.conf import settings
from transformers import LlamaForCausalLM, LlamaTokenizer,T5Tokenizer, T5ForConditionalGeneration
from functools import lru_cache
import re
from django.utils.timezone import now
from django.db.models import Max
from django.utils.text import slugify
import httpx
import asyncio
from random import sample
from django.db.models import Avg
from asgiref.sync import async_to_sync
from django.contrib.auth import logout
from django.core.exceptions import ObjectDoesNotExist


logger = logging.getLogger(__name__)
def recommended_books(request):
    local_recommendations, open_library_books = get_hybrid_recommendations(request.user)
    return render(request, 'recommendations.html', {
        'local_recommendations': local_recommendations,
        'open_library_books': open_library_books,
    })


async def fetch_open_library_data(book_titles):
    from urllib.parse import quote

    urls = [f"https://openlibrary.org/search.json?title={quote(title)}" for title in book_titles]

    async def fetch_with_retry(client, url, retries=3, delay=2):
        for attempt in range(retries):
            try:
                response = await client.get(url, timeout=10)
                if response.status_code == 200:
                    return response.json()
                logger.warning(f"Unexpected status code {response.status_code} for {url}")
            except httpx.RequestError as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            await asyncio.sleep(delay)
        logger.error(f"All retries failed for {url}")
        return None

    async with httpx.AsyncClient() as client:
        tasks = [fetch_with_retry(client, url) for url in urls]
        responses = await asyncio.gather(*tasks)

    return responses


def home(request):
    categories = Category.objects.all()
    book_ids = list(Book.objects.values_list('id', flat=True))
    random_ids = sample(book_ids, min(len(book_ids), 5)) if book_ids else []
    featured_books = Book.objects.filter(id__in=random_ids)
    popular_books = Book.objects.annotate(
        avg_rating=Avg('reviews__star_rating')
    ).order_by('-avg_rating')[:5]

    recently_viewed = []
    profile = None
    featured_books_with_api = []
    local_recommendations = []
    open_library_books = []

    if request.user.is_authenticated:
        profile = get_object_or_404(UserProfile, user=request.user)
        local_recommendations, open_library_books = get_hybrid_recommendations(request.user)
        recently_viewed = Book.objects.filter(
            id__in=UserActivity.objects.filter(
                user=request.user, activity_type='view'
            ).order_by('-timestamp').values_list('book_id', flat=True)
        ).distinct()

    try:
        book_titles = [book.title for book in featured_books]
        open_library_data = async_to_sync(fetch_open_library_data)(book_titles)
        for book, api_data in zip(featured_books, open_library_data):
            book_api_details = {
                'key': api_data.get('docs', [{}])[0].get('key', book.id),
                'title': book.title,
                'author': book.author,
                'publish_year': api_data.get('docs', [{}])[0].get('first_publish_year', 'Unknown Year'),
                'cover_url': f"https://covers.openlibrary.org/b/id/{api_data.get('docs', [{}])[0].get('cover_i', '')}-L.jpg"
                if api_data.get('docs') else None,
            }
            featured_books_with_api.append((book, book_api_details))
            print(featured_books_with_api)

    except Exception as e:
        logger.error(f"Error fetching Open Library data: {e}")
        featured_books_with_api = [(book, {'key': book.id}) for book in featured_books]

    context = {
        'categories': categories,
        'featured_books_with_api': featured_books_with_api,
        'popular_books': popular_books,
        'recently_viewed': recently_viewed,
        'profile': profile,
        'local_recommendations': local_recommendations,
        'open_library_books': open_library_books,
    }
    return render(request, 'home.html', context)


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)  
        if form.is_valid():
            user = form.save()  
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)  
            login(request, user)  
            return redirect('dashboard')  
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form}) 

@login_required
def dashboard_view(request):
    user = request.user
    try:
        profile = user.userprofile
    except ObjectDoesNotExist:
        profile = None

    profile_picture_url = (
        profile.profile_picture.url if profile and profile.profile_picture else None
    )
    if not request.user.is_authenticated:
        return redirect('login')  
    profile = get_object_or_404(UserProfile, user=request.user)
    recent_reviews = Review.objects.filter(user=request.user).order_by('-created_at')[:5]
    recent_bookmarks = Bookmark.objects.filter(user=request.user).order_by('-id')[:5]
    recent_progress = BookProgress.objects.filter(user=request.user).order_by('-id')[:5]
    recent_reviews = [{"type": "review", "activity": review} for review in recent_reviews]
    recent_bookmarks = [{"type": "bookmark", "activity": bookmark} for bookmark in recent_bookmarks]
    recent_progress = [{"type": "progress", "activity": progress} for progress in recent_progress]
    activities = sorted(
        recent_reviews + recent_bookmarks + recent_progress,
        key=lambda x: getattr(x["activity"], "created_at", getattr(x["activity"], "id", 0)),
        reverse=True
    )[:10]  
    context = {
        'profile_picture_url': profile_picture_url,
        'profile': profile,
        'user': user,
        'activities': activities,
    }
    return render(request, 'dashboard.html', context)

@lru_cache(maxsize=1)
def load_t5_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text_by_tokens(text, tokenizer, max_tokens=512, overlap=50):
    tokens = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        if len(chunk) > 0:
            chunks.append(chunk)
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_chunk(chunk, model, tokenizer, max_length=300, min_length=50):
    input_text = f"summarize: {chunk}"  
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=2,
        num_beams=4,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_pdf_with_gptj(pdf_path):
    try:
        tokenizer, model = load_t5_model()
        reader = PdfReader(pdf_path)
        full_text = clean_text(" ".join(page.extract_text() or "" for page in reader.pages))
        chunks = chunk_text_by_tokens(full_text, tokenizer)
        summaries = [summarize_chunk(chunk, model, tokenizer) for chunk in chunks]
        return " ".join(summaries)
    except Exception as e:
        logger.exception(f"Error during summarization: {e}")
        raise ValueError(f"Error summarizing PDF: {e}")
    
def summarize_book(request, book_id):
    if request.method == 'POST':
        book = get_object_or_404(Book, id=book_id)
        pdf_path = book.file.path
        try:
            if not os.path.isfile(pdf_path):
                logger.error(f"PDF file does not exist: {pdf_path}")
                messages.error(request, "The PDF file does not exist.")
                return redirect('book_detail', book_id=book_id)
            logger.info(f"Starting summarization for Book ID: {book_id}, file: {pdf_path}")
            new_summary = summarize_pdf_with_gptj(pdf_path)
            logger.info(f"Summary generated successfully for Book ID: {book_id}")
            return render(request, 'summary_result.html', {
                'book': book,
                'new_summary': new_summary,
            })
        except ValueError as ve:
            logger.warning(f"ValueError during summarization: {ve}")
            messages.error(request, "Unable to summarize the book. Please check the file and try again.")
            return redirect('book_detail', book_id=book_id)
        except Exception as e:
            logger.exception(f"Unexpected error during summarization: {e}")
            messages.error(request, "An unexpected error occurred. Please try again later.")
            return redirect('book_detail', book_id=book_id)
        
class BookRecommendationView(ListView):
    model = Book
    template_name = 'recommendations.html'
    context_object_name = 'recommendations'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.user.is_authenticated:
            user_profile = self.request.user.userprofile
            preferred_categories = user_profile.preferred_categories.all()
            if preferred_categories.exists():
                context['recommendations'] = Book.objects.filter(
                    categories__in=preferred_categories
                ).distinct()[:10]
            else:
                context['recommendations'] = Book.objects.order_by('?')[:10]
        else:
            context['recommendations'] = Book.objects.order_by('?')[:10]
        return context
    
@login_required
def progress_tracker(request):
    books = BookProgress.objects.filter(user=request.user)
    return render(request, 'progress_tracker.html', {'books': books})

def search_books(request):
    query = request.GET.get('q', '')  
    local_results = []
    api_results = []
    if query:  
        local_results = Book.objects.filter(title__icontains=query)
        url = f"https://openlibrary.org/search.json?q={query}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for book in data.get("docs", []):
                # Optimization: Use data directly from search results to avoid N+1 API calls
                cover_i = book.get("cover_i")
                api_results.append({
                    "key": book.get("key"),
                    "title": book.get("title", "Unknown Title"),
                    "author": ", ".join(book.get("author_name", [])),
                    "cover_url": f"https://covers.openlibrary.org/b/id/{cover_i}-M.jpg" if cover_i else None,
                    "read_link": None, # Read link requires extra call, simplified for performance
                })
    context = {
        'query': query,
        'local_results': local_results,
        'api_results': api_results,
    }
    return render(request, 'search_books.html', context)

@login_required
def get_book_summary(request, book_id):
    book = Book.objects.get(id=book_id)
    summary = book.summarize()  
    return JsonResponse({'summary': summary})
nlp = spacy.load('en_core_web_sm')

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

@api_view(['POST'])
def sentiment_analysis_view(request, book_id):
    try:
        book = Book.objects.get(id=book_id)
        metadata = f"{book.title} {book.author} {book.file}"
        blob = TextBlob(metadata)
        sentiment_score = blob.sentiment.polarity
        book.sentiment = sentiment_score
        book.save()
        return JsonResponse({"message": "Sentiment analyzed and saved.", "sentiment": sentiment_score})
    except Book.DoesNotExist:
        return JsonResponse({"error": "Book not found."}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
def get_books(request):
    books = list(Book.objects.values('id', 'title'))
    return JsonResponse(books, safe=False)

def get_book_suggestions(request):
    query = request.GET.get('query', '')
    if not query:
        return JsonResponse([], safe=False)
    try:
        response = requests.get(f'https://archive.org/advancedsearch.php?q={query}&output=json', timeout=10)
        response.raise_for_status()  
        data = response.json()
        if "response" not in data or "docs" not in data["response"]:
            return JsonResponse({"error": "Invalid response format"}, status=500)
        suggestions = [
            {"title": doc.get("title", "Unknown Title")}
            for doc in data["response"].get("docs", [])
        ]
        return JsonResponse(suggestions, safe=False)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)
    except ValueError:
        return JsonResponse({"error": "Failed to parse JSON response"}, status=500)
    
def fetch_book_content(download_link):
    try:
        response = requests.get(download_link, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
def upload_book(request):
    if request.method == 'POST':
        form = BookUploadForm(request.POST,request.FILES)
        if form.is_valid():
            book = form.save()
            if book.file.name.endswith('.pdf'):
                with open(book.file.path,'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()
                book.description = text
                book.save()
            return redirect('book_list')
    else:
        form = BookUploadForm()
    return render(request,'upload_book.html', {'form': form})


def add_review(request, book_id=None, key=None):
    if request.method == "POST":
        star_rating = request.POST.get("star_rating")
        review_text = request.POST.get("review_text")
        if not star_rating or not review_text:
            messages.error(request, "Please provide a rating and review text.")
            return redirect(request.META.get('HTTP_REFERER', '/'))
        try:
            if book_id:
                book = get_object_or_404(Book, id=book_id)
                Review.objects.create(book=book, user=request.user, star_rating=star_rating, review_text=review_text)
            elif key:
                Review.objects.create(
                    book_key=key,
                    user=request.user,
                    star_rating=star_rating,
                    review_text=review_text,
                )
            else:
                messages.error(request, "Invalid request.")
                return redirect(request.META.get('HTTP_REFERER', '/'))
            messages.success(request, "Review submitted successfully!")
        except Exception as e:
            messages.error(request, f"An error occurred: {e}")
        return redirect(request.META.get('HTTP_REFERER', '/'))
    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

def summarize_openlibrary(request, key):
    try:
        sanitized_key = key.replace('_', '/')  
        logger.debug(f"Sanitized Open Library key: {sanitized_key}")
        api_url = f"https://openlibrary.org/{sanitized_key}.json"
        response = requests.get(api_url)
        response.raise_for_status()
        book_data = response.json()
        title = book_data.get("title", "Unknown Title")
        
        # Handle description as string or dict
        description = book_data.get("description", "No description available")
        if isinstance(description, dict):
            description = description.get("value", "No description available")
        
        # Construct summary
        summary = f"Summary for '{title}': {description[:200]}{'...' if len(description) > 200 else ''}"
        
        # Return data to template
        context = {"title": title, "description": description, "summary": summary}
        return render(request, "openlibrary_summary.html", context)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Open Library book details: {e}")
        return render(request, "error.html", {"error": "Failed to fetch book details"})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return render(request, "error.html", {"error": "An unexpected error occurred"})

    
def book_detail(request, book_id=None, key=None):
    book_details = {}
    reviews = []
    pdf_url = None
    summarize_url = None
    generated_summary = None
    local_book = None  # Initialize local_book
    logger = logging.getLogger(__name__)

    try:
        if book_id:  
            local_book = get_object_or_404(Book, id=book_id)
            pdf_url = f"/api/book/{book_id}/pdf/" if local_book.file else None
            summarize_url = reverse('summarize_book', args=[book_id])
            book_details = {
                "book_id": local_book.id,
                "title": local_book.title,
                "author": local_book.author,
                "cover_url": local_book.cover_url,
                "description": local_book.summary,
            }
            reviews = local_book.reviews.select_related('user').all()
            if request.user.is_authenticated:
                activity = UserActivity.objects.create(
                    user=request.user, 
                    book=local_book, 
                    activity_type='view'
                )
                print(f"Activity logged: {activity}")

        elif key:  
            key = key.strip("/")
            if not re.match(r"^[a-zA-Z0-9/_-]+$", key):
                logger.error(f"Invalid key provided: {key}")
                return HttpResponse("Invalid key format.", status=400)
            api_url = f"https://openlibrary.org/{key}.json"
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            authors = [
                author_response.get("name", "Unknown Author")
                for author in data.get("authors", [])
                if (author_response := requests.get(f"https://openlibrary.org{author['author']['key']}.json").json())
            ]
            book_details = {
                "key": key,
                "title": data.get("title", "Unknown Title"),
                "author": ", ".join(authors),
                "publish_year": data.get("publish_date", "Unknown Year"),
                "cover_url": f"https://covers.openlibrary.org/b/id/{data.get('covers', [None])[0]}-L.jpg",
                "description": data.get("description", "No description available"),
            }
            summarize_url = reverse('summarize_openlibrary', args=[key])
            reviews = Review.objects.filter(book__key=key).select_related('user')
            if request.user.is_authenticated:
                UserActivity.objects.create(user=request.user, open_library_key=key, activity_type='detail_click')
        else:
            return HttpResponse("Invalid request. Either book_id or key must be provided.", status=400)

        if request.method == "POST" and "generate_summary" in request.POST:
            response = requests.post(summarize_url, timeout=10)
            response.raise_for_status()
            generated_summary = response.json().get("summary", "Summary could not be generated.")
    except requests.RequestException as e:
        logger.error(f"Error processing request: {e}")
        return HttpResponse("An error occurred while fetching book details. Please try again.", status=500)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return HttpResponse("An unexpected error occurred. Please try again.", status=500)

    average_rating = reviews.aggregate(Avg('star_rating'))['star_rating__avg'] or 0
    context = {
        "book": book_details,
        "book_id": local_book.id if local_book else None,  # Avoid referencing undefined local_book
        "reviews": reviews,
        "average_rating": round(average_rating, 1),
        "pdf_url": pdf_url,
        "summarize_url": summarize_url,
        "generated_summary": generated_summary,
    }
    return render(request, "book_detail.html", context)


def book_list(request):
    books = Book.objects.values('id', 'title')  
    return JsonResponse(list(books), safe=False)
class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer

def category_books(request, category_name):
    api_url = f"https://openlibrary.org/subjects/{category_name.lower()}.json"
    try:
        response = requests.get(api_url)
        response.raise_for_status()  
        data = response.json()
        book_list = []
        for book in data.get("works", []):
            read_url = None
            work_id = book.get('key', '').split('/')[-1]
            if work_id:
                try:
                    work_api_url = f"https://openlibrary.org/works/{work_id}.json"
                    work_response = requests.get(work_api_url)
                    work_response.raise_for_status()
                    work_data = work_response.json()
                    if 'availability' in work_data:
                        read_url = work_data['availability'].get('read_url', None)
                except requests.RequestException as e:
                    print(f"Error fetching read_url for work_id {work_id}: {e}")
            book_list.append({
                "title": book.get("title", "Unknown Title"),
                "author": ", ".join(author.get("name", "Unknown Author") for author in book.get("authors", [])),
                "key": book.get("key"),
                "cover_url": f"https://covers.openlibrary.org/b/id/{book.get('cover_id')}-M.jpg" if book.get("cover_id") else "/static/images/no_cover_available.jpg",
                "read_url": read_url,
            })
    except requests.RequestException as e:
        print(f"Error fetching data from OpenLibrary API: {e}")
        book_list = []  
    return render(request, "category_books.html", {"book_list": book_list})

@login_required
def user_profile(request, username):
    user = get_object_or_404(User, username=username)
    return render(request, 'user_profile.html', {'user': user})

@csrf_exempt
def user_activities(request):
    activities = UserActivity.objects.all().order_by('-timestamp')[:10]  
    activities_data = [
        {
            'id': activity.id,
            'activity_type': activity.activity_type,
            'timestamp': activity.timestamp,
        }
        for activity in activities
    ]
    return JsonResponse(activities_data, safe=False)

def csrf_token_view(request):
    csrf_token = get_token(request)
    response = JsonResponse({"csrfToken": csrf_token})
    response["Access-Control-Allow-Credentials"] = "true"
    return response

@ensure_csrf_cookie
def serve_react_app(request):
    return render(request, 'index.html')

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_info(request):
    if request.user.is_authenticated:
        serializer = UserSerializer(request.user)
        return Response(serializer.data, status=status.HTTP_200_OK)
    return Response({"error": "User not authenticated"}, status=status.HTTP_401_UNAUTHORIZED)

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST, request.FILES)
        if form.is_valid():
            user=form.save()
            profile, created = UserProfile.objects.get_or_create(user=user)
            if created:
                profile.profile_picture = request.FILES.get('profile_picture')
                profile.save()
            messages.success(request, 'Your account has been created!')
            return redirect('login')
           
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})

@ensure_csrf_cookie
def get_csrf_token(request):
    return JsonResponse({"message": "CSRF token set"})
class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
class UserActivityViewSet(viewsets.ModelViewSet):
    queryset = UserActivity.objects.all()
    serializer_class = UserActivitySerializer
class CustomLoginView(LoginView):
    def get_success_url(self):
        user_id = self.request.user.id
        return resolve_url('dashboard', user_id=user_id)
    
def view_pdf(request, book_id):
    book = get_object_or_404(Book, id=book_id)
    if book.file:
        pdf_path = book.file.path
        if os.path.exists(pdf_path):
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = f'inline; filename="{book.title}.pdf"'
            with open(pdf_path, 'rb') as pdf:
                response.write(pdf.read())
            return response
    return HttpResponse("No PDF available for this book.", status=404)

def summarize_pdf_view(request):
    if request.method == 'POST' and request.FILES['pdf']:
        pdf_file = request.FILES['pdf']
        pdf_path = default_storage.save(pdf_file.name, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        summary = summarize_text(text)
        return render(request, 'summary.html', {'summary': summary})
    return render(request, 'upload.html')
def all_books_view(request):
    books = Book.objects.all()  
    return render(request, 'all_books.html', {'books': books})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def rate_book(request, book_id):
    try:
        star_rating = int(request.data.get('star_rating', 0))  
        if 1 <= star_rating <= 5:  
            book = get_object_or_404(Book, id=book_id)
            review, created = Review.objects.get_or_create(
                user=request.user,
                book=book,
                defaults={'star_rating': star_rating}
            )
            if not created:
                review.star_rating = star_rating
                review.save()
            reviews = book.reviews.all()
            if reviews.exists():
                avg_rating = reviews.aggregate(Avg('star_rating'))['star_rating__avg']
                book.sentiment = avg_rating  
                book.save()
            return JsonResponse({'message': 'Rating submitted successfully.'})
        else:
            return JsonResponse({'error': 'Invalid rating. Must be between 1 and 5.'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def log_activity(request):
    if request.method == 'POST':
        if request.user.is_authenticated:
            try:
                # Parse and validate JSON data
                data = json.loads(request.body)
                activity_type = data.get('activity_type')
                open_library_key = data.get('open_library_key')

                if not activity_type or not open_library_key:
                    return JsonResponse({'error': 'Invalid data. "activity_type" and "open_library_key" are required.'}, status=400)

                # Log the activity
                UserActivity.objects.create(user=request.user, activity_type=activity_type, open_library_key=open_library_key)
                return JsonResponse({'message': 'Activity logged successfully'}, status=201)

            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
            except Exception as e:
                return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)
        else:
            return JsonResponse({'error': 'User is not authenticated'}, status=401)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


def log_user_view(user, book):
    UserActivity.objects.update_or_create(
        user=user, 
        book=book, 
        activity_type='view',
        defaults={'timestamp': now()}
    )

def get_recently_viewed_books(user, limit=10):
    recent_activities = UserActivity.objects.filter(
        user=user, 
        activity_type='view'
    ).values('book').annotate(latest_view=Max('timestamp')).order_by('-latest_view')
    
    book_ids = [entry['book'] for entry in recent_activities[:limit]]
    return Book.objects.filter(id__in=book_ids)

@login_required
def toggle_bookmark(request, book_id):
    if request.method == "POST":
        book = get_object_or_404(Book, id=book_id)
        bookmark, created = Bookmark.objects.get_or_create(user=request.user, book=book)
        if not created:
            bookmark.delete()
            return JsonResponse({'status': 'removed'})
        return JsonResponse({'status': 'added'})
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def toggle_like(request, book_id):
    book = get_object_or_404(Book, id=book_id)
    if request.user in book.liked_by.all():
        book.liked_by.remove(request.user)
        return JsonResponse({'status': 'unliked'})
    else:
        book.liked_by.add(request.user)
        return JsonResponse({'status': 'liked'})
    
from django.contrib.auth.decorators import login_required

@login_required
def user_profile(request):
    liked_books = request.user.liked_books.all()
    
    bookmarks = Bookmark.objects.filter(user=request.user).select_related('book')
    profile = get_object_or_404(UserProfile, user=request.user)
    context = {
        'liked_books': liked_books,
        'bookmarks': bookmarks,
        'profile': profile,
    }
    return render(request, 'user_profile.html', context)

@login_required
def edit_profile(request):
    user = request.user
    try:
        profile = user.userprofile  
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=user)

    if request.method == 'POST':
        user_form = UserEditForm(request.POST, instance=user)
        profile_form = UserProfileForm(request.POST, request.FILES, instance=profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save(user=request.user)
            messages.success(request, "Your profile has been updated.")
            return redirect('dashboard')
    else:
        user_form = UserEditForm(instance=user)
        profile_form = UserProfileForm(instance=profile)

    context = {
        'user_form': user_form,
        'profile_form': profile_form,
    }
    return render(request, 'edit_profile.html', context)

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('home')
    else:
        return HttpResponse('Method Not Allowed', status=405)