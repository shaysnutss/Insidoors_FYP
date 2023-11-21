// package com.service.accountservice;

// import org.springframework.boot.test.context.TestConfiguration;
// import org.springframework.context.annotation.Bean;
// import org.testcontainers.containers.MySQLContainer;
// import org.testcontainers.utility.DockerImageName;

// @TestConfiguration
// public class DatabaseTestConfiguration {
    
//     @Bean
//     public MySQLContainer<?> mySQLContainer() {
//         MySQLContainer<?> container = new MySQLContainer<>(DockerImageName.parse("mysql:8.0.26"))
//                 .withDatabaseName("account")
//                 .withUsername("test")
//                 .withPassword("test");
//         container.start();
//         return container;
//     }
// }
