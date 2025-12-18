from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models import Avg, UniqueConstraint
from django.db.models.signals import post_save, post_delete
from django.core.validators import FileExtensionValidator, MinValueValidator, MaxValueValidator
import fitz  
from textblob import TextBlob
from celery import shared_task
import logging

logger = logging.getLogger(__name__)

def calculate_sentiment(text):
    if not text:
        return 0  
    return TextBlob(text).sentiment.polarity

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    def __str__(self):
        return self.name
    
class Book(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    embedding_vector = models.JSONField(null=True, blank=True)
    file = models.FileField(
        upload_to='pdfs/',
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])],
        null=True,
        blank=True
    )
    liked_by = models.ManyToManyField(User, through='Like', related_name='liked_books')
    key = models.CharField(max_length=100, null=True, blank=True)
    categories = models.ManyToManyField(Category, related_name='books')
    sentiment = models.FloatField(default=0.0)
    is_recommended = models.BooleanField(default=False)
    content = models.TextField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    cover_url = models.URLField(max_length=500, null=True, blank=True)
    page_count = models.PositiveIntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.title} by {self.author}"
    
    def save(self, *args, **kwargs):
        if self.file and not self.content:
            process_pdf_content.delay(self.id)
            self.processed_content = True
        super().save(*args, **kwargs)
        # Trigger embedding update asynchronously or directly
        # To avoid circular imports, we import here
        try:
            from .recommendation import update_book_embedding
            # Only update if essential fields changed or no embedding exists.
            # detailed check omitted for brevity/performance
            update_book_embedding(self)
        except ImportError:
            pass # recommendation module might not be fully ready during migrations
        except Exception as e:
            logger.error(f"Failed to update embedding for book {self.id}: {e}")

    def summarize(self):
        return self.content[:100] + '...' if self.content and len(self.content) > 100 else self.content
    
    def update_sentiment_from_reviews(self):
        reviews = self.reviews.all()
        self.sentiment = reviews.aggregate(avg_sentiment=Avg('sentiment_score'))['avg_sentiment'] or 0
        self.is_recommended = self.sentiment > 0
        self.save(update_fields=['sentiment','is_recommended'])

    def save_open_library_book(key, title, author, cover_url=None):
        book, created = Book.objects.get_or_create(
            key=key,
            defaults={
                'title': title,
                'author': author,
                'cover_url': cover_url,
            }
        )
        return book
    
    @property
    def like_count(self):
        return self.likes.count()

    
@shared_task
def process_pdf_content(book_id):
    try:
        book = Book.objects.get(id=book_id)
        content = ""
        page_count = 0
        with fitz.open(book.file.path) as doc:
            for page in doc:
                content += page.get_text("text")
                page_count += 1
        book.content = content[:10000]  
        book.page_count = page_count
        book.save(update_fields=['content', 'page_count'])
    except Exception as e:
        print(f"Error processing PDF for Book ID {book_id}: {e}")

class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='reviews')
    review_text = models.TextField(blank=True, null=True)
    openlibrary_key = models.CharField(max_length=255, null=True, blank=True)  
    review_text = models.TextField(blank=True, null=True)
    book_key = models.CharField(max_length=255, null=True, blank=True)
    sentiment_score = models.FloatField(default=0.0)
    star_rating = models.PositiveSmallIntegerField(
        default=3,
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.review_text:
            sentiment_mapping = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
            self.sentiment_score = sentiment_mapping.get(self.star_rating, 0.0)
        else:
            self.sentiment_score = calculate_sentiment(self.review_text)
        super().save(*args, **kwargs)

    class Meta:
        constraints = [
            UniqueConstraint(fields=['user', 'book','openlibrary_key'], name='unique_user_book_review')
        ]

@receiver(post_save, sender=Review)
def update_book_sentiment_on_save(sender, instance, **kwargs):
    if instance.book:
        instance.book.update_sentiment_from_reviews()

@receiver(post_delete, sender=Review)
def update_book_sentiment_on_delete(sender, instance, **kwargs):
    if instance.book:
        instance.book.update_sentiment_from_reviews()

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    favorite_books = models.ManyToManyField(Book, blank=True, related_name='favored_by')
    location = models.CharField(max_length=100, blank=True, null=True)
    preferred_categories = models.ManyToManyField(Category, blank=True)
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
    @receiver(post_save, sender=User)
    def create_or_save_user_profile(sender, instance, created, **kwargs):
        if created:
            UserProfile.objects.create(user=instance)
        else:
            instance.userprofile.save()

class Bookmark(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    location = models.CharField(max_length=100)
    note = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"{self.user.username} - {self.book.title} @ {self.location}"
    
class Like(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name='book_likes')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['user', 'book'], name='unique_user_book_like'),
        ]

    def __str__(self):
        return f"{self.user.username} liked {self.book.title}"
    
class BookProgress(models.Model):
    STATUS_CHOICES = [
        ('not_started', 'Not Started'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='not_started')

    def __str__(self):
        return f"{self.book.title} - {self.status}"
    
class UserActivity(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, null=True, blank=True, on_delete=models.CASCADE)
    open_library_key = models.CharField(max_length=255, null=True, blank=True)
    activity_type = models.CharField(max_length=50)
    weight = models.FloatField(default=1.0)
    timestamp = models.DateTimeField(auto_now_add=True)
    additional_data = models.JSONField(null=True, blank=True)


    def __str__(self):
        return f"{self.user.username} - {self.activity_type} at {self.timestamp}"
    
class OpenLibraryBook(models.Model):
    key = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255, null=True, blank=True)
    cover_url = models.URLField(max_length=500, null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)
