create schema behavioral_analysis;
use behavioral_analysis;
drop table if exists behavioral_analysis_service;
create table behavioral_analysis_service (
    ba_id int AUTO_INCREMENT primary key,
    employee_id int,
    risk_rating int,
    suspected_cases int
);