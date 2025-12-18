from django.urls import path,re_path
from django.contrib.auth import views as auth_views
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.home, name='home'),  
    path("category/<str:category_name>/", views.category_books, name="category_books"),
    path('category/<int:category_id>/', views.category_books, name='category_books'),  
    path('search/', views.search_books, name='search_books'),  
    path('book/<int:book_id>/summary/', views.get_book_summary, name='get_book_summary'),  
    path('books/', views.all_books_view, name='all_books'),  
    path('register/', views.register, name='register'),  
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),  
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),  
    path('upload/', views.upload_book, name='upload_book'),  
    path('recommendations/', views.BookRecommendationView.as_view(), name='recommendations'),  
    path('progress/', views.progress_tracker, name='progress_tracker'),  
    path('book/<int:book_id>/pdf/', views.view_pdf, name='view_pdf'),  
    path('summarize/', views.summarize_pdf_view, name='summarize_pdf'),  
    path('profile/', views.user_profile, name='user_profile'),
    path('rate-book/<int:book_id>/', views.rate_book, name='rate_book'),
    path('recommended/',views.recommended_books, name='recommended_books'),
    path('api/book/<int:book_id>/summarize/', views.summarize_book, name='summarize_book'),
    path('book/<int:book_id>/', views.book_detail, name='book_detail'),  
    re_path(r'^api/openlibrary/summarize/(?P<key>.+)/$', views.summarize_openlibrary, name='summarize_openlibrary'),
    path('api/books/', views.book_list, name='api_book_list'),  
    path('book/local/<int:book_id>/', views.book_detail, name='local_book_detail'),
    re_path(r'^api/books/(?P<key>.+)/?$', views.book_detail, name='api_book_detail'),
    path('api/profile/<str:username>/', views.user_profile, name='user_profile'),
    re_path(r'^api/openlibrary/(?P<key>.+)/add_review/$', views.add_review, name='add_openlibrary_review'),
    path('api/book/<int:book_id>/add_review/', views.add_review, name='add_review'),
    path('api/useractivities/', views.user_activities, name='api_user_activities'),
    path('bookmark/<int:book_id>/', views.toggle_bookmark, name='bookmark'),
     path('log_activity/', views.log_activity, name='log_activity'),
    path('like/<int:book_id>/', views.toggle_like, name='like'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
]