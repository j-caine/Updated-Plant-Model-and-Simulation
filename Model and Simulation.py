import pandas as pd
import numpy as np
from typing import List, Dict, Any
import random

import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# DATA PROCESSOR
# ---------------------------------------------------------------------------
class DataProcessor:
    """Loads and preprocesses CSV data for plants, pollinators, and weather."""

    def load_plant_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        return df

    def load_bees_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        return df

    def load_butterflies_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        return df

    def load_weather_data(self, csv_path: str) -> pd.DataFrame:
        wdf = pd.read_csv(csv_path, parse_dates=["DATE"])
        wdf = wdf.copy()
        wdf["DayOfYear"] = wdf["DATE"].dt.dayofyear
        wdf["PRCP"] = wdf["PRCP"].fillna(0.0)
        wdf["AWND"] = wdf["AWND"].fillna(0.0)
        return wdf

# ---------------------------------------------------------------------------
# HELPER: Distances, Competition, etc.
# ---------------------------------------------------------------------------
def distance(p1, p2):
    """Euclidean distance between two (x,y) coords."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_competition_factor(this_plant: Dict[str, Any],
                               all_plants: List[Dict[str,Any]],
                               dist_threshold=20.0) -> float:
    """
    If two plants are near each other (distance < dist_threshold),
    the one with the more negative competition effect 'wins'.

    If this_plant is itself a strong competitor (very negative effect), it overshadows others
    but is not overshadowed. So we check if there's any neighbor with a more negative effect
    than this_plant. If so, we hamper this_plant's factor by 0.5 for each such overshadow.

    """
    x0, y0 = this_plant["pos"]
    my_comp = this_plant["competition_effect"]

    overshadow_count = 0
    for p in all_plants:
        if p is not this_plant and p["biomass"] > 0.1:
            # check distance
            d = distance((x0,y0), p["pos"])
            if d < dist_threshold:
                # see who is more negative
                their_comp = p["competition_effect"]
                # If the other plant has a strictly *more negative* competition value,
                # that means they overshadow me => hamper my growth
                if their_comp < my_comp:
                    overshadow_count += 1

    # Each overshadow roughly halves the daily increment
    factor = 1.0 * (0.5**overshadow_count)
    # But ensure we never drop below 0.1
    if factor < 0.1:
        factor = 0.1
    return factor


# ---------------------------------------------------------------------------
# PLANT GROWTH MODEL
# ---------------------------------------------------------------------------
class PlantGrowthModel:
    """Uses GDD + logistic growth, ensuring no daily increment = 0 (we clamp a min)."""

    def __init__(self, base_temp=50.0):
        self.base_temp = base_temp

    def update_biomass(self,
                       plant_state: Dict[str, Any],
                       tmax: float,
                       tmin: float,
                       prcp: float,
                       comp_factor: float) -> None:
        growth_rate = plant_state["growth_rate"]  # 0..5
        water_use   = plant_state["water_use"]    # 1..3
        current_biomass = plant_state["biomass"]
        height_scale    = plant_state["height_scale"]  # 0..5

        # GDD
        avg_temp = (tmax + tmin)/2.0
        gdd = max(0.0, avg_temp - self.base_temp)

        # Water factor
        water_factor = 1.0
        if water_use == 3 and prcp < 0.1:
            water_factor = 0.5
        elif water_use == 1 and prcp < 0.01:
            water_factor = 0.3

        # logistic
        r = growth_rate * 0.05
        K = 5.0 * height_scale
        if K < 0.1:
            # no capacity
            return

        gdd_factor = gdd/10.0
        dB = r * current_biomass * (1.0 - current_biomass/K)
        dB *= water_factor
        dB *= gdd_factor
        dB *= comp_factor

        # ensure no daily increment is 0, if dB < 1e-6
        if dB < 1e-6:
            dB = 0.0001

        plant_state["biomass"] = min(K, current_biomass + dB)  # clamp to K

# ---------------------------------------------------------------------------
# POLLINATION NETWORK MODEL
# ---------------------------------------------------------------------------
class PollinationNetworkModel:
    """Manages pollinator -> color matching + active months."""

    def __init__(self):
        self.pollinators = {}

    def add_pollinator(self, poll_name: str, color_list: List[str],
                       months_range: str,
                       rarity: int, poll_benefit: int) -> None:
        start_m, end_m = self._parse_month_range(months_range)
        self.pollinators[poll_name] = {
            "colors": color_list,
            "start_month": start_m,
            "end_month": end_m,
            "rarity": rarity,
            "poll_benefit": poll_benefit
        }

    def pollinate(self, date, plants_in_bloom: List[Dict[str, Any]]) -> Dict[str, str]:
        visits = {}
        m = date.month
        for poll_name, info in self.pollinators.items():
            if info["start_month"] <= m <= info["end_month"]:
                # active
                color_matches = []
                for p in plants_in_bloom:
                    pollset = set([c.strip().lower() for c in info["colors"]])
                    plantset= set([c.strip().lower() for c in p["bloom_color"].split(",")])
                    if pollset.intersection(plantset):
                        color_matches.append(p)
                if color_matches:
                    chosen = color_matches[0]
                    visits[poll_name] = chosen["name"]
                else:
                    visits[poll_name] = None
            else:
                visits[poll_name] = None
        return visits

    def _parse_month_range(self, s:str)->(int,int):
        mm = {
            "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
            "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
        }
        parts = s.split("-")
        if len(parts)==2:
            start_m=mm.get(parts[0],1)
            end_m=mm.get(parts[1],12)
        else:
            start_m=1
            end_m=12
        return(start_m,end_m)

# ---------------------------------------------------------------------------
# SIMULATION ENGINE
# ---------------------------------------------------------------------------
class SimulationEngine:
    """
    - Assign random (x,y) to each plant
    - Each day: update growth, compute overshadow factor, do pollination
    - For the "sprouting dots," we store them in self.child_positions
    - For pollinators, we store positions that move day to day
    """

    def __init__(self,
                 plant_df: pd.DataFrame,
                 weather_df: pd.DataFrame,
                 pollinators: pd.DataFrame,
                 pollination_model: PollinationNetworkModel,
                 growth_model: PlantGrowthModel,
                 time_step: str = "daily"):
        self.plant_df = plant_df.copy()
        self.weather_df = weather_df.copy()
        self.pollinators_df = pollinators.copy()
        self.pollination_model = pollination_model
        self.growth_model = growth_model
        self.time_step = time_step

        # Build internal plant states
        self.plants_state = []
        # randomly place plants in e.g. 100 x 100 plane
        for _, row in self.plant_df.iterrows():
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)

            pstate = {
                "symbol": row["Symbol"],
                "name": row["Name"],
                "growth_rate": row["Growth Rate"],
                "competition_effect": row["Competition Effect"],
                "water_use": row["Water Use"],
                "light_requirement": row["Light Requirement"],
                "pollinator_dependency": row["Pollinator Dependency"],
                "bloom_time_str": str(row["Bloom Time"]),
                "bloom_color": str(row["Bloom color"]),
                "height_scale": row["Height Scale (0-5)"],
                "biomass": 0.01,  # start with small > 0
                "is_flowering": False,
                "pos": (x,y)  # store x,y
            }
            self.plants_state.append(pstate)

        # parse weather date, etc.
        self.weather_df.sort_values("DATE", inplace=True)
        self.weather_df.reset_index(drop=True, inplace=True)

        # For "child dots," store them in a list: (date, x, y, plantName)
        self.child_positions = []

        # For pollinators (like bees/butterflies),
        # We'll store a dictionary poll_name->(x,y).
        # We'll randomly place them at day=0
        self.pollinator_positions = {}
        for name in self.pollination_model.pollinators:
            self.pollinator_positions[name] = (random.uniform(0,100), random.uniform(0,100))

        # We'll store daily "pollinator trails" as (date, poll_name, x1, y1, x2, y2)
        self.pollinator_trails = []

    def run_simulation(self) -> pd.DataFrame:
        results = []
        for idx, day in self.weather_df.iterrows():
            date = day["DATE"]
            tmax = day["TMAX"]
            tmin = day["TMIN"]
            prcp = day["PRCP"]

            # 1) Growth & bloom
            for plant in self.plants_state:
                # overshadow factor
                overshadow_factor = compute_competition_factor(plant, self.plants_state)
                self.growth_model.update_biomass(plant, tmax, tmin, prcp, overshadow_factor)

                # check bloom
                is_bloom = self._check_if_blooming(date, plant["bloom_time_str"])
                plant["is_flowering"] = is_bloom

                # If biomass increased, sprout new dots
                if plant["biomass"] > 1.0:
                    cx = plant["pos"][0] + random.uniform(-2,2)
                    cy = plant["pos"][1] + random.uniform(-2,2)
                    self.child_positions.append((date, cx, cy, plant["name"]))

            # 2) pollination
            blooming_plants = [p for p in self.plants_state if p["is_flowering"]]
            visits = self.pollination_model.pollinate(date, blooming_plants)

            # 3) Move pollinators between visited plants
            # If a pollinator visited a plant, we "animate" a line from current to that plant
            for poll_name, visited_plant in visits.items():
                old_pos = self.pollinator_positions[poll_name]
                if visited_plant is not None:
                    # find that plant's pos
                    target_p = next(x for x in self.plants_state if x["name"]==visited_plant)
                    new_pos = target_p["pos"]
                    # record a "trail"
                    self.pollinator_trails.append((date, poll_name,
                                                   old_pos[0], old_pos[1],
                                                   new_pos[0], new_pos[1]))
                    # pollinator now moves to new_pos
                    self.pollinator_positions[poll_name] = new_pos
                # else do no move

            # 4) record daily biomass
            day_record = {"Date": date}
            for plant in self.plants_state:
                day_record[plant["name"]] = plant["biomass"]
            results.append(day_record)

        return pd.DataFrame(results)


    def _check_if_blooming(self, date, bloom_time_str:str)->bool:
        month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                     "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        parts = bloom_time_str.split("-")
        if len(parts)==2:
            s_m=month_map.get(parts[0],1)
            e_m=month_map.get(parts[1],12)
            return s_m<=date.month<=e_m
        else:
            return True


# ---------------------------------------------------------------------------
# VISUALIZER
# ---------------------------------------------------------------------------
class Visualizer:
    def display_results(self, daily_df: pd.DataFrame,
                        final_plants_df: pd.DataFrame,
                        engine: 'SimulationEngine'):
        """
          1) print preview
          2) line chart of daily biomass
          3) final 2D scatter of plants, child dots, pollinator trails
        """
        print("\n=== Daily Simulation Results (first 10 days) ===")
        print(daily_df.head(10))

        print("\n=== Final Plant States ===")
        print(final_plants_df[["name","biomass","is_flowering","pos"]])

        # 1) Multi-line chart of daily biomass
        melt_df = daily_df.melt(id_vars=["Date"], var_name="Plant", value_name="Biomass",
                                value_vars=[c for c in daily_df.columns if c!="Date"])
        fig1 = px.line(melt_df, x="Date", y="Biomass", color="Plant",
                title="Daily Biomass (Competition, no zero growth)")
        fig1.show()

        # 2) 2D scatter:
        #   - parent plants: big circle
        #   - child dots: small circle
        #   - pollinator trails: lines
        # We'll create a Plotly figure in "scatter" mode + add lines.

        # gather all plant positions
        x_plant = []
        y_plant = []
        names_plant = []
        for p in engine.plants_state:
            x_plant.append(p["pos"][0])
            y_plant.append(p["pos"][1])
            names_plant.append(p["name"])

        # gather child dots
        # child_positions is list of (date, x, y, plantName)
        x_child = [c[1] for c in engine.child_positions]
        y_child = [c[2] for c in engine.child_positions]
        child_label= [c[3] for c in engine.child_positions]

        fig2 = go.Figure()

        # base plane
        fig2.add_trace(go.Scatter(
            x=x_plant, y=y_plant,
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle'),
            name='Parent Plants',
            text=names_plant
        ))
        if x_child:
            fig2.add_trace(go.Scatter(
                x=x_child, y=y_child,
                mode='markers',
                marker=dict(size=5, color='green', symbol='circle-open'),
                name='Child Dots',
                text=child_label
            ))

        # pollinator trails => lines
        # pollinator_trails is list of (date, poll_name, x1,y1, x2,y2)
        for (d, poll, x1,y1,x2,y2) in engine.pollinator_trails:
            fig2.add_trace(go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode='lines+markers',
                line=dict(dash='dash', color='red'),
                marker=dict(symbol='triangle-up', color='red'),
                name=f'{poll} on {d.strftime("%m/%d")}'
            ))

        fig2.update_layout(
            title='Final 2D Scatter: Plants, Children, Pollinator Trails',
            showlegend=True
        )
        fig2.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # 1) Data
    dp = DataProcessor()

    plants_df = dp.load_plant_data("plants.csv")
    bees_df   = dp.load_bees_data("bees.csv")
    butter_df = dp.load_butterflies_data("butterflies.csv")
    weather_df= dp.load_weather_data("weather.csv")

    pollination_model = PollinationNetworkModel()

    # add pollinators
    for _, row in bees_df.iterrows():
        poll_name = row["Bees"]
        color_list= [c.strip() for c in row["Colors Attracted"].split(",")]
        months_str= row["Months Active"]
        rarity    = row["Rarity"]
        poll_ben  = row["Pollinator Benefit"]
        pollination_model.add_pollinator(poll_name, color_list, months_str, rarity, poll_ben)

    for _, row in butter_df.iterrows():
        poll_name= row["Butterfly"]
        color_list= [c.strip() for c in row["Colors Attracted"].split(",")]
        months_str= row["Months Active"]
        rarity    = row["Rarity"]
        poll_ben  = row["Pollinator Benefit"]
        pollination_model.add_pollinator(poll_name, color_list, months_str, rarity, poll_ben)

    growth_model = PlantGrowthModel(base_temp=50.0)

    engine = SimulationEngine(
        plant_df=plants_df,
        weather_df=weather_df,
        pollinators=pd.concat([bees_df,butter_df]),
        pollination_model=pollination_model,
        growth_model=growth_model,
        time_step="daily"
    )

    daily_df = engine.run_simulation()
    final_plants_df = pd.DataFrame(engine.plants_state)

    vis = Visualizer()
    vis.display_results(daily_df, final_plants_df, engine)

if __name__=="__main__":
    main()
