
from pydantic import BaseModel, Field
from typing import Annotated, Literal


class InputCheck(BaseModel):
    area: Annotated[float, Field(..., gt=0, description="Area of property")]
    bedrooms: Annotated[float, Field(..., gt=0, description="Number of bedrooms")]
    bathrooms: Annotated[float, Field(..., gt=0, description="Number of bathrooms")]
    status: Annotated[Literal["Ready to Move", "Under Construction"], Field(..., description="Construction status of the property")]
    neworold: Annotated[Literal["Resale", "New Property"], Field(..., description="Whether the property is new or resale")]
    furnished_status: Annotated[Literal[0, 1, 2], Field(..., description="0: Unfurnished, 1: Semi-Furnished, 2: Fully Furnished")]
    type_of_building: Annotated[Literal["Flat", "Individual House"], Field(..., description="Type of property")]
    category: Annotated[Literal["North Mumbai", "East Mumbai", "West Mumbai", "South Mumbai"], Field(..., description="Geographical region of the property in Mumbai")]
