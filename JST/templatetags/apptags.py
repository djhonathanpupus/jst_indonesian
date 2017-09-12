'''
Created on 8 Jul 2017

@author: djhonathanpupus
'''
from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter
def add(key, number):
    return int(key) + int(number)

@register.filter
def get_range(value):
    value = int(value)
    return range(0, value)