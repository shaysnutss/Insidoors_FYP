// package com.service.accountservice;

// import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
// import com.fasterxml.jackson.core.JsonProcessingException;
// import com.fasterxml.jackson.databind.ObjectMapper;
// import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
// import com.service.accountservice.model.Account;
// import com.service.accountservice.repository.AccountServiceRepository;
// import com.zaxxer.hikari.HikariDataSource;

// import org.junit.jupiter.api.AfterAll;
// import org.junit.jupiter.api.BeforeAll;
// import org.junit.jupiter.api.MethodOrderer;
// import org.junit.jupiter.api.Order;
// import org.junit.jupiter.api.Test;
// import org.junit.jupiter.api.TestMethodOrder;
// import org.springframework.beans.factory.annotation.Autowired;
// import org.springframework.boot.test.context.SpringBootTest;
// import org.springframework.boot.test.web.client.TestRestTemplate;
// import org.springframework.boot.test.web.server.LocalServerPort;
// import org.springframework.context.annotation.Import;
// import org.springframework.core.ParameterizedTypeReference;
// import org.springframework.http.*;
// import org.springframework.test.context.DynamicPropertyRegistry;
// import org.springframework.test.context.DynamicPropertySource;
// import org.springframework.test.context.TestPropertySource;
// import org.springframework.test.context.jdbc.Sql;
// import org.testcontainers.containers.MySQLContainer;
// import org.testcontainers.containers.wait.strategy.Wait;
// import org.testcontainers.junit.jupiter.*;
// import org.testcontainers.utility.DockerImageName;

// import jakarta.activation.DataSource;

// import java.util.*;

// import static org.junit.jupiter.api.Assertions.assertEquals;
// import static org.junit.jupiter.api.Assertions.assertNotNull;

// @SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
// @TestPropertySource(locations = "classpath:application-test.properties")
// @Testcontainers
// @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
// @Import(TestConfig.class)
// public class AccountServiceIntegrationTests {

//     private static DataSource dataSource;
    
//     @LocalServerPort
//     private int port;

//     @Container
//     public static MySQLContainer<?> mySQLContainer = new MySQLContainer<>(DockerImageName.parse("mysql:8.0.26"))
//         .withDatabaseName("account")
//         .withUsername("test")
//         .withPassword("test")
//         .waitingFor(Wait.forListeningPort())
//         .withEnv("MYSQL_ROOT_HOST", "%");

//     @DynamicPropertySource
//     static void registerPgProperties(DynamicPropertyRegistry registry) {
//         registry.add("spring.datasource.url", mySQLContainer::getJdbcUrl);
//         registry.add("spring.datasource.password", mySQLContainer::getPassword);
//         registry.add("spring.datasource.username", mySQLContainer::getUsername);
//     }

//     @Autowired
//     private TestRestTemplate restTemplate;

//     @Autowired
//     private AccountServiceRepository accServiceRepo;

//     private static HttpHeaders headers;

//     private ObjectMapper objectMapper = new ObjectMapper();

//     @BeforeAll
//     public static void init() {
//         headers = new HttpHeaders();
//         headers.setContentType(MediaType.APPLICATION_JSON);
//     }

//     @AfterAll
//     static void tearDown() {
//         if (dataSource instanceof HikariDataSource) {
//             ((HikariDataSource) dataSource).close();
//         }
//     }

//     private String createURLWithPort() {
//         return "http://localhost:" + port + "/api/v1/account";
//     }

//     @Test
//     @Order(1)
//     @Sql(statements = "INSERT INTO account (id, name, email, password, role) values (1, \"Henry\", \"Henry@gmail.com\", \"8283467\", \"SOC\")", executionPhase = Sql.ExecutionPhase.BEFORE_TEST_METHOD)
//     public void testAccountList() {
//         HttpEntity<String> entity = new HttpEntity<>(null, headers);
//         ResponseEntity<List<Account>> response = restTemplate.exchange(
//             (createURLWithPort() + "/getAllAccounts"), HttpMethod.GET, entity, new ParameterizedTypeReference<List<Account>>(){});
//         List<Account> accounts = response.getBody();
//         assertNotNull(accounts);
//         assertEquals(response.getStatusCode(), HttpStatus.OK);
//         assertEquals(accounts.size(), accServiceRepo.findAll().size());
//         assertEquals(1, accServiceRepo.findAll().size());
//     }

//     @Test
//     @Order(2)
//     public void testAccountById() {
//         HttpEntity<String> entity = new HttpEntity<>(null, headers);
//         ResponseEntity<Account> response = restTemplate.exchange(
//             (createURLWithPort() + "/getAccountById/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<Account>(){});
//         Account account = response.getBody();

//         assertEquals(response.getStatusCode(), HttpStatus.OK);
//         assertEquals(account, Account.accountNoPassword(accServiceRepo.findById(1L).get()));
//         assertEquals("Henry", accServiceRepo.findById(1L).get().getName());
//     }

//     @Test
//     @Order(3)
//     public void testAccountByEmail() {
//         HttpEntity<String> entity = new HttpEntity<>(null, headers);
//         ResponseEntity<Account> response = restTemplate.exchange(
//             (createURLWithPort() + "/getAccountByEmail/Henry@gmail.com"), HttpMethod.GET, entity, new ParameterizedTypeReference<Account>(){});
//         Account account = response.getBody();

//         assertEquals(response.getStatusCode(), HttpStatus.OK);
//         assertEquals(account, Account.accountNoPassword(accServiceRepo.findByEmail("Henry@gmail.com")));
//         assertEquals("Henry", accServiceRepo.findByEmail("Henry@gmail.com").getName());
//     }

//     @Test
//     @Order(4)
//     public void testNameById() {
//         HttpEntity<String> entity = new HttpEntity<>(null, headers);
//         ResponseEntity<String> response = restTemplate.exchange(
//             (createURLWithPort() + "/getNameById/1"), HttpMethod.GET, entity, new ParameterizedTypeReference<String>(){});
//         String name = response.getBody();

//         assertEquals(response.getStatusCode(), HttpStatus.OK);
//         assertEquals(name, accServiceRepo.findByEmail("Henry@gmail.com").getName());
//     }
// }
