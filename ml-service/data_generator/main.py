from sqlalchemy.orm import sessionmaker
from datetime import datetime, date, time, timedelta
import random
import names
from models import engine, Base
from models import Employee, Building_Access, PC_Access, Proxy_Log
import csv

"""
suspect case 1 - after hour login
suspect case 2 - impossible traveller or potential account sharing (no building access, but pc logs)
suspect case 3 - terminated employee login
suspect case 4 - failed attempt to enter building / potential tailgating
suspect case 5 - impossible traveller
suspect case 6 - potential data exfiltration

user profile 1 --> 9am - 6pm Mon-Fri, 
user profile 2 --> 9am - 6pm any day 
user profile 3 --> 12 hour shift any day (8am-8pm or 8pm-8am)
"""

NUM_OF_EMPLOYEES = 2000
START_DATE = date(2023, 1, 1)
END_DATE = date(2023, 1, 31)

bizunit_list = ['Wealth Management', 'Technology & Ops', 'Investment Bank', 'Asset Management', 'Retail Banking']
status_list = ['Sensitive', 'Non-sensitive', 'Non-sensitive', 'Non-sensitive']
profile_list = ['1', '1', '1', '1', '2', '2', '3']
location_list = ['Singapore', 'Zurich', 'Hong Kong', 'Sydney', 'London', 'New York', 'New Jersey', 'Beijing', 'Tokyo',
                 'Seoul']

Session = sessionmaker(bind=engine)
session = Session()
Base.metadata.create_all(engine)


def generate_employees():
    gender = 'Male'
    for x in range(NUM_OF_EMPLOYEES//2):
        gender = 'Female' if gender == 'Male' else 'Male'
        firstname = names.get_first_name(gender=gender)
        lastname = names.get_last_name()
        email = firstname + "." + lastname + "@abc.com"
        joined_date = datetime.strptime('{} {}'.format(random.randint(1, 366), random.randrange(1995, 2022, 1)),
                                        '%j %Y')
        terminated_date = None
        business_unit = random.choice(bizunit_list)
        profile = random.choice(profile_list)
        location = random.choice(location_list)
        suspect = 0
        if (random.randint(0, 30)) == 0:  # 1/50 chance to have a terminated date append terminated date
            terminated_date = joined_date + timedelta(days=random.randrange(60, 3600, 5))

        employee = Employee(gender=gender, firstname=firstname, lastname=lastname, email=email, joined_date=joined_date,
                            business_unit=business_unit, terminated_date=terminated_date,
                            profile=profile, location=location, suspect=suspect)
        session.add(employee)

    session.commit()


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def normal_login_time(current_date):
    # 8-10am
    time_str = str(random.randrange(8, 10, 1)).rjust(2, '0') + ":" + str(random.randrange(00, 59, 1)).rjust(2,
                                                                                                            '0') + ":" + str(
        random.randrange(00, 59, 1)).rjust(2, '0')
    return datetime.combine(current_date, datetime.strptime(time_str, '%H:%M:%S').time())


def late_login_time(current_date):
    # 7-9pm
    time_str = str(random.randrange(19, 21, 1)) + ":" + str(random.randrange(00, 59, 1)).rjust(2, '0') + ":" + str(
        random.randrange(00, 59, 1)).rjust(2, '0')
    return datetime.combine(current_date, datetime.strptime(time_str, '%H:%M:%S').time())


def normal_logoff_time(current_date):
    # 5-7pm
    time_str = str(random.randrange(17, 19, 1)) + ":" + str(random.randrange(00, 59, 1)).rjust(2, '0') + ":" + str(
        random.randrange(00, 59, 1)).rjust(2, '0')
    return datetime.combine(current_date, datetime.strptime(time_str, '%H:%M:%S').time())


def odd_hour_login_time(current_date):
    # if weekend, any time login, if not weekend, only add hours
    if current_date.strftime("%A") != "Saturday" and current_date.strftime("%A") != "Sunday":
        hour_list = ['0', '1', '2', '3', '4', '5', '21', '22', '23']
        logon_hour = random.choice(hour_list)
    else:
        logon_hour = random.randrange(1, 23, 1)
    time_str = str(logon_hour).rjust(2, '0') + ":" + str(random.randrange(00, 59, 1)).rjust(2, '0') + ":" + str(
        random.randrange(00, 59, 1)).rjust(2, '0')
    return datetime.combine(current_date, datetime.strptime(time_str, '%H:%M:%S').time())


def odd_hour_login_time_ex_weekend(current_date):
    # odd hour timing
    hour_list = ['0', '1', '2', '3', '4', '5', '21', '22', '23']
    logon_hour = random.choice(hour_list)
    time_str = str(logon_hour).rjust(2, '0') + ":" + str(random.randrange(00, 59, 1)).rjust(2, '0') + ":" + str(
        random.randrange(00, 59, 1)).rjust(2, '0')
    return datetime.combine(current_date, datetime.strptime(time_str, '%H:%M:%S').time())


def logout_anytime(login_datetime, hours):
    return login_datetime + timedelta(hours=hours, minutes=random.randrange(1, 59, 1))


def login_anytime(current_date):
    # anytime in 24hour
    time_str = str(random.randrange(1, 23, 1)).rjust(2, '0') + ":" + str(random.randrange(00, 59, 1)).rjust(2,
                                                                                                            '0') + ":" + str(
        random.randrange(00, 59, 1)).rjust(2, '0')
    return datetime.combine(current_date, datetime.strptime(time_str, '%H:%M:%S').time())


def generate_pc_access_log():
    for employee in session.query(Employee).order_by(Employee.id):
        # loop daily between START and END date
        for current_date in date_range(START_DATE, END_DATE):
            curr_day = current_date.strftime("%A")
            terminated_on_curr_day = "YES"
            # check for this current day if employee is terminated or not
            if employee.terminated_date is None or (
                    employee.terminated_date is not None and employee.terminated_date > current_date):
                terminated_on_curr_day = "NO"

            # Mon-Fri 9-6 and not terminated employee
            if employee.profile == 1 and terminated_on_curr_day == "NO":
                # normal login/logoff - introduce randomness to simulate people on leave
                if curr_day != "Saturday" and curr_day != "Sunday" and (random.randint(0, 35)) != 0:
                    pc_access = PC_Access(user_id=employee.id, access_date_time=normal_login_time(current_date),
                                          log_on_off="Log On", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=0)
                    session.add(pc_access)

                    pc_access = PC_Access(user_id=employee.id, access_date_time=normal_logoff_time(current_date),
                                          log_on_off="Log Off", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=0)
                    session.add(pc_access)

                # suspect case 1 - after hour login for profile 1
                if (random.randint(0, 400)) == 0:
                    pc_access = PC_Access(user_id=employee.id, access_date_time=odd_hour_login_time(current_date),
                                          log_on_off="Log On", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=1)
                    session.add(pc_access)

                    pc_access = PC_Access(user_id=employee.id, access_date_time=odd_hour_login_time(current_date),
                                          log_on_off="Log Off", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=1)
                    session.add(pc_access)
                    employee.suspect = True
                    session.add(employee)

            # 9am-6pm any day and not terminated employee
            if employee.profile == 2 and terminated_on_curr_day == "NO":
                # normal login/logoff 20% chance employee is on off day
                if (random.randint(0, 4)) != 0:
                    pc_access = PC_Access(user_id=employee.id, access_date_time=normal_login_time(current_date),
                                          log_on_off="Log On", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=0)
                    session.add(pc_access)

                    pc_access = PC_Access(user_id=employee.id, access_date_time=normal_logoff_time(current_date),
                                          log_on_off="Log Off", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=0)
                    session.add(pc_access)

                # suspect case 1 - after hour login for profile 2
                if (random.randint(0, 400)) == 0:
                    login_datetime = odd_hour_login_time_ex_weekend(current_date)
                    pc_access = PC_Access(user_id=employee.id, access_date_time=login_datetime,
                                          log_on_off="Log On", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=1)
                    session.add(pc_access)

                    # logout after 1-10 hours randomly
                    pc_access = PC_Access(user_id=employee.id,
                                          access_date_time=logout_anytime(login_datetime, random.randrange(1, 10, 1)),
                                          log_on_off="Log Off", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=1)
                    session.add(pc_access)
                    employee.suspect = True
                    session.add(employee)

            # 8am-8pm rotating and not terminated employee
            if employee.profile == 3 and terminated_on_curr_day == "NO":
                # 25% chance employee starts at 8am, 25% chance employee starts at 8pm, 50% chance employee is on rest day
                workornot = (random.randint(0, 3))
                if workornot == 0:
                    login_datetime = normal_login_time(current_date)
                elif workornot == 1:
                    login_datetime = late_login_time(current_date)
                if workornot == 0 or workornot == 1:
                    pc_access = PC_Access(user_id=employee.id, access_date_time=login_datetime,
                                          log_on_off="Log On", machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=0)
                    session.add(pc_access)
                    logout_datetime = logout_anytime(login_datetime, random.randrange(11, 13, 1))
                    pc_access = PC_Access(user_id=employee.id, access_date_time=logout_datetime, log_on_off="Log Off",
                                          machine_name="PC_" + str(employee.id),
                                          machine_location=employee.location, suspect=0)
                    session.add(pc_access)

                    # suspect case 1 - after hour login for profile 3
                    # log in again after shit is over within 1-4 hours and stay for 1-2 hours
                    if (random.randint(0, 400)) == 0:
                        login_datetime = logout_anytime(logout_datetime, random.randrange(1, 4, 1))
                        logout_datetime = logout_anytime(login_datetime, random.randrange(1, 2, 1))
                        pc_access = PC_Access(user_id=employee.id, access_date_time=login_datetime, log_on_off="Log On",
                                              machine_name="PC_" + str(employee.id),
                                              machine_location=employee.location, suspect=1)
                        session.add(pc_access)
                        pc_access = PC_Access(user_id=employee.id, access_date_time=logout_datetime,
                                              log_on_off="Log Off", machine_name="PC_" + str(employee.id),
                                              machine_location=employee.location, suspect=1)
                        session.add(pc_access)
                        employee.suspect = True
                        session.add(employee)

            # suspect case 2-impossible traveller
            elif terminated_on_curr_day == "NO" and (random.randint(0, 500)) == 0:
                location2 = random.choice(location_list)
                while location2 == employee.location:  # get a different second location
                    location2 = random.choice(location_list)

                PC = "PC_" + str(employee.id + random.randrange(NUM_OF_EMPLOYEES, NUM_OF_EMPLOYEES * 2, 1))
                login_datetime = login_anytime(current_date)
                pc_access = PC_Access(user_id=employee.id, access_date_time=login_datetime,
                                      log_on_off="Log On", machine_name=PC,
                                      machine_location=location2, suspect=2)
                session.add(pc_access)

                pc_access = PC_Access(user_id=employee.id,
                                      access_date_time=logout_anytime(login_datetime, random.randrange(1, 10, 1)),
                                      log_on_off="Log Off", machine_name=PC,
                                      machine_location=location2, suspect=2)
                session.add(pc_access)
                employee.suspect = True
                session.add(employee)

            # terminated employee and 1 in 150 chance
            # suspect case 3 - terminated employee login
            elif terminated_on_curr_day == "YES" and (random.randint(0, 800)) == 0:
                login_datetime = login_anytime(current_date)
                pc_access = PC_Access(user_id=employee.id, access_date_time=login_datetime,
                                      log_on_off="Log On", machine_name="PC_" + str(employee.id),
                                      machine_location=employee.location, suspect=3)
                session.add(pc_access)

                pc_access = PC_Access(user_id=employee.id,
                                      access_date_time=logout_anytime(login_datetime, random.randrange(1, 10, 1)),
                                      log_on_off="Log Off", machine_name="PC_" + str(employee.id),
                                      machine_location=employee.location, suspect=3)
                session.add(pc_access)
                employee.suspect = True
                session.add(employee)

    session.commit()


def generate_building_access_log():
    for pc_access in session.query(PC_Access).order_by(PC_Access.id):
        if pc_access.log_on_off == "Log On":
            building_access_timestamp = pc_access.access_date_time - timedelta(minutes=random.randrange(3, 15))
            direction = "IN"
        else:
            building_access_timestamp = pc_access.access_date_time + timedelta(minutes=random.randrange(3, 15))
            direction = "OUT"

        status = "SUCCESS"
        suspect = 0
        # suspect case 4 - failed attempt to enter building / potential tailgating
        if pc_access.suspect != 0 and pc_access.suspect != 2 and (random.randint(0, 5)) == 0:
            status = "FAIL"
            suspect = 4

        # suspect case 3 - terminated employee
        if pc_access.suspect == 3:
            suspect = 3

        # suspect case 5 - impossible traveller
        if pc_access.suspect == 2:
            suspect = 2
            if (random.randint(0, 5)) == 0:
                suspect = 5
                pc_access.suspect = 5
                session.add(pc_access)
                session.commit()
            # suspect case 2 - potential account sharing - no trace of building access but pc log is there
            else:
                continue

        building_access = Building_Access(user_id=pc_access.user_id, access_date_time=building_access_timestamp,
                                          direction=direction, status=status,
                                          office_location=pc_access.machine_location,
                                          suspect=suspect)
        session.add(building_access)
    session.commit()


def generate_proxy_log():
    with open('urls.csv', 'r') as f:
        urls = list(csv.reader(f, delimiter=','))

    for pc_access in session.query(PC_Access).order_by(PC_Access.id):
        # to write proxy logs
        if pc_access.log_on_off == "Log On":
            # randomly access up to 10 times within 10 mins to 6 hours after login
            url_access_times = random.randint(0, 10)
            for i in range(url_access_times):
                urlno = random.randint(0, 750)  # pick random url from urls.csv
                proxy_log_timestamp = pc_access.access_date_time + timedelta(minutes=random.randrange(10, 360))

                bytes_in = random.randint(10000, 9000000)
                bytes_out = random.randint(10000, 9000000)
                suspect = 0
                # suspect case 6 - potential data exfiltration
                if random.randint(0, 50) == 0 and pc_access.suspect != 0:
                    #employee already a suspect
                    bytes_out = random.randint(10000000, 1000000000)
                    suspect = 6
                elif random.randint(0, 500) == 0 and pc_access.suspect == 0:
                    #employee is not a suspect yet
                    bytes_out = random.randint(10000000, 1000000000)
                    suspect = 6
                    employee = session.query(Employee).filter(Employee.id == pc_access.user_id).one()
                    employee.suspect = True
                    session.add(employee)

                proxy_log = Proxy_Log(user_id=pc_access.user_id, access_date_time=proxy_log_timestamp,
                                      machine_name=pc_access.machine_name,
                                      url=urls[urlno][0], category=urls[urlno][1], bytes_in=bytes_in,
                                      bytes_out=bytes_out, suspect=suspect)
                session.add(proxy_log)
    session.commit()


def main():
    generate_employees()
    generate_pc_access_log()
    generate_building_access_log()
    generate_proxy_log()


if __name__ == "__main__":
    main()
