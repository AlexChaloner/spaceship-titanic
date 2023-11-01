
class TitanicDataRow:
    passenger_id: str
    home_planet: str
    cryo_sleep: bool
    cabin: (str, str, str)
    destination: str
    age: float
    vip: bool
    room_service: float
    food_court: float
    shopping_mall: float
    spa: float
    vr_deck: float
    name: str
    transported: bool

    def __init__(self, csv_dictionary_row: dict):
        self.passenger_id = csv_dictionary_row.get("PassengerId", "")
        self.home_planet = csv_dictionary_row.get("HomePlanet", "")
        self.cryo_sleep = csv_dictionary_row.get("CryoSleep", False) if csv_dictionary_row.get("CryoSleep", "") != "" else False
        cabin_value = csv_dictionary_row.get("Cabin", "")
        if cabin_value:
            cabin_parts = cabin_value.split('/')
            self.cabin = (cabin_parts[0], cabin_parts[1], cabin_parts[2]) if len(cabin_parts) == 3 else ("", "", "")
        else:
            self.cabin = ("", "", "")
        self.destination = csv_dictionary_row.get("Destination", "")
        self.age = float(csv_dictionary_row.get("Age", 0.0)) if csv_dictionary_row.get("Age", "") != "" else 0.0
        self.vip = csv_dictionary_row.get("VIP", False) if csv_dictionary_row.get("VIP", "") != "" else False
        self.room_service = float(csv_dictionary_row.get("RoomService", 0.0)) if csv_dictionary_row.get("RoomService", "") != "" else 0.0
        self.food_court = float(csv_dictionary_row.get("FoodCourt", 0.0)) if csv_dictionary_row.get("FoodCourt", "") != "" else 0.0
        self.shopping_mall = float(csv_dictionary_row.get("ShoppingMall", 0.0)) if csv_dictionary_row.get("ShoppingMall", "") != "" else 0.0
        self.spa = float(csv_dictionary_row.get("Spa", 0.0)) if csv_dictionary_row.get("Spa", "") != "" else 0.0
        self.vr_deck = float(csv_dictionary_row.get("VRDeck", 0.0)) if csv_dictionary_row.get("VRDeck", "") != "" else 0.0
        self.name = csv_dictionary_row.get("Name", "")
        transported = csv_dictionary_row.get("Transported", "")
        self.transported = True if transported == "True" else False


class TitanicData:
    data: list[TitanicDataRow]

    def __init__(self, csv_dictionary_rows: list[dict]):
        self.data = [TitanicDataRow(row) for row in csv_dictionary_rows]
