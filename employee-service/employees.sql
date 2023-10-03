create schema employees;
use employees;
drop table if exists employees;
create table employees (
    id int AUTO_INCREMENT primary key,
    firstname varchar(255),
    lastname varchar(255),
    email varchar(255),
    gender varchar(255),
    business_unit varchar(255),
    joined_date date,
    terminated_date date,
    location varchar(255)
);