from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Bob"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0.0, lt=10.0, default=5.0, description="A decimal number between 0.0 and 10.0 representing the student's CGPA")

new_student = {"age": "22", "email": "abc@gmail.com", "cgpa": 1.0}

student = Student(**new_student)

student_dict = dict(student)

student_json= student.model_dump_json()

print(student_dict)
print(student_json)