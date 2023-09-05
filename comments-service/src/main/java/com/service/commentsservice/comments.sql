create schema comments;

drop table if exists `comments_service`;
create table `comments_service` (
    `comment_id` int AUTO_INCREMENT primary key,
    `comment_description` varchar(2083),
    `task_management_id` int,
    `account_id` int

);