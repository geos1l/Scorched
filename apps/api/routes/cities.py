from fastapi import APIRouter

router = APIRouter()

CITIES = [{"city_id": "toronto", "name": "Toronto"}]


@router.get("/cities")
def get_cities():
    return {"cities": CITIES}
