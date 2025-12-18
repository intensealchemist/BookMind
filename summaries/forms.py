from django import forms 
from .models import Book
from django.contrib.auth.models import User
from .models import UserProfile
from django.contrib.auth.forms import UserCreationForm
from django.db import IntegrityError

class UserEditForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    profile_picture = forms.ImageField(required=False)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2', 'profile_picture']

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()
            profile_picture = self.cleaned_data.get('profile_picture')
            try:
                UserProfile.objects.get_or_create(
                    user=user,
                    defaults={'profile_picture': profile_picture}
                )
            except IntegrityError:
                pass

        return user


class BookUploadForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ['title', 'author', 'file', 'categories']
        widgets = {
            'categories': forms.CheckboxSelectMultiple(),
        }

    def clean_file(self):
        file = self.cleaned_data.get('file', False)
        if file:
            if not file.name.endswith('.pdf'):
                raise forms.ValidationError("Only PDF files are allowed.")
        return file
    
class UserProfileForm(forms.ModelForm):
    email = forms.EmailField(required=False)

    class Meta:
        model = UserProfile
        fields = ['profile_picture','bio']

    def save(self, user, commit=True):
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        profile = super().save(commit=commit)
        profile.bio = self.cleaned_data.get('bio')
        if commit:
            profile.save()
        return profile

