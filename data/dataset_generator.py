"""
This script is to generate synthetic data, with the Data Generating Process, right now it is completely random and
inspired by the Synthetic Data Generator of Digilab which can be found here
https://gitlab.com/digilab.overheid.nl/ecosystem/synthetic-data-generator/sdg-realm-seeder
"""
import pandas as pd
from data import constants
import random
from faker import Faker
from datetime import datetime, timedelta, timezone
fake = Faker('nl_NL')

random.seed(69420)
amount_datapoints = 10000

def generate_synthetic_persons():
    person_ids = list(range(1,amount_datapoints + 1))
    person_birth_place_of_births = [random.choice(constants.MUNICIPALITIES) for _ in range(amount_datapoints)]
    # shuffle ensures no duplicates are found, but this takes very long
    # person_birth_national_insurance_number = [random.shuffle(list(range(10000000, 999999999)))[0:amount_datapoints]]
    person_birth_national_insurance_number = [random.randint(10000000, 999999999) for _ in range(amount_datapoints)]
    person_birth_genders = [random.choice(constants.GENDERS) for _ in range(amount_datapoints)]
    # gender is too random right now, this is nothing like the true distribution
    person_naming_first_names = []
    for gender in person_birth_genders:
        if gender == 'M':
            value = random.choice(constants.GIVEN_NAMES_MALE)
        elif gender == 'F':
            value = random.choice(constants.GIVEN_NAMES_FEMALE)
        else:
            value = random.choice(constants.GIVEN_NAMES_X)
        person_naming_first_names.append(value)
    person_naming_last_names = [random.choice(constants.SURNAMES) for _ in range(amount_datapoints)]
    # person_addressregistration_address =
    # person_addressregistration_municipality =
    # Generate inception of persons between now and 32 days ago
    person_inceptions = [datetime.now(timezone.utc) -
                         timedelta(seconds=random.randint(1, int(timedelta(days=32).total_seconds())))
                         for _ in range(amount_datapoints)]

    persons = pd.DataFrame({
        'id': person_ids,
        'place_of_birth': person_birth_place_of_births,
        'national_insurance_number': person_birth_national_insurance_number,
        'gender': person_birth_genders,
        'first_name': person_naming_first_names,
        'last_name': person_naming_last_names,
        'municipality': person_birth_place_of_births,
        'inception': person_inceptions
    })
    print(persons)


def generate_synthetic_addresses():
    address_ids = list(range(1, amount_datapoints + 1))
    address_registration_streets = [random.choice(constants.STREET_NAMES) for _ in range(amount_datapoints)]
    address_registration_house_numbers = [random.randint(1, 100) for _ in range(amount_datapoints)]
    address_registration_house_number_additions = [random.choice(constants.HOUSE_NUMBER_ADDITIONS)
                                                   for _ in range(amount_datapoints)]
    address_registration_postcodes = ['{0}{1}{2}'.format(random.randint(1000, 10000),
                                                         random.choice(constants.POSTCODE_ADDITIONS),
                                                         random.choice(constants.POSTCODE_ADDITIONS))
                                      for _ in range(amount_datapoints)]
    address_registration_municipalities = [random.choice(constants.MUNICIPALITIES) for _ in range(amount_datapoints)]
    # municipalities does not have the same postal code as postcodes
    address_registration_purposes = [random.choice(constants.PURPOSES) for _ in range(amount_datapoints)]
    address_registration_surfaces = [random.randint(20, 200) for _ in range(amount_datapoints)]

    addresses = pd.DataFrame({
        'id': address_ids,
        'street': address_registration_streets,
        'house_number': address_registration_house_numbers,
        'house_number_additions': address_registration_house_number_additions,
        'postcode': address_registration_postcodes,
        'municipality': address_registration_municipalities,
        'purpose': address_registration_purposes,
        'surface': address_registration_surfaces,
    })
    print(addresses)


def generate_synthetic_buildings():
    building_ids = list(range(1, amount_datapoints + 1))
    # for now there is just 1 building on 1 address
    building_address_assignment = [random.shuffle(list(range(1, amount_datapoints + 1)))]
    # Generate inception of buildings between now and 32 days ago
    building_registration_constructed_ats = [datetime.now(timezone.utc) -
                                             timedelta(seconds=random.randint(1,
                                                                              int(timedelta(days=32).total_seconds())))
                                             for _ in range(amount_datapoints)]
    building_registration_surfaces = [random.randint(20, 200) for _ in range(amount_datapoints)]
    building_types = [random.choice(["Residential", "NonResidential"]) for _ in range(amount_datapoints)]

    building_valuation_values = [random.randint(100000, 1000000) for _ in range(amount_datapoints)]
    building_valuation_valuation_ats = [datetime.now(timezone.utc) -
                                             timedelta(seconds=random.randint(1,
                                                                              int(timedelta(days=6).total_seconds())))
                                             for _ in range(amount_datapoints)]
    building_valuation_effective_ats = [datetime.now(timezone.utc) -
                                             timedelta(seconds=random.randint(1,
                                                                              int(timedelta(days=8).total_seconds())))
                                             for _ in range(amount_datapoints)]
    #
    # building_subject_ninos =
    # building_subject_rsins =
    #
    # building_valuedetermination_ids =
    # building_valuedetermination_addressable_object_ids =
    # building_valuedetermination_owners =
    # building_valuedetermination_occupants =
    # building_valuedetermination_registered_persons =
    # building_valuedetermination_type =
    # building_valuedetermination_values =
    # building_valuedetermination_municipality =

    buildings = pd.DataFrame({
        'id': building_ids,
        'address_assignment': building_address_assignment,
        'registration_constructed_at': building_registration_constructed_ats,
        'registration_surface': building_registration_surfaces,
        'value_determination'
        'type': building_types,

    })

    print(buildings)

def generate_synthetic_vehicles():
    vehicle_ids = list(range(1, amount_datapoints + 1))
    vehicle_registration_car_models = [random.choice(constants.CAR_MODELS) for _ in range(amount_datapoints)]
    vehicle_registration_number_plates = [fake.license_plate() for _ in range(amount_datapoints)]

    vehicles = pd.DataFrame({
        'id': vehicle_ids,
        'car_model': vehicle_registration_car_models,
        'number_plate': vehicle_registration_number_plates,
    })
    print(vehicles)
