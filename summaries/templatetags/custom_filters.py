from django import template

register = template.Library()

@register.filter
def zip_lists(a, b):
    """Zips two lists together."""
    return zip(a, b)
