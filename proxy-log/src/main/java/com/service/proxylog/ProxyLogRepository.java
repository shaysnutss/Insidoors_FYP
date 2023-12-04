package com.service.proxylog;

import java.util.*;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

@Repository
public interface ProxyLogRepository extends JpaRepository<ProxyLog, Long> {
    
    @Query(value = "SELECT * FROM proxy_log WHERE ID > ?1", nativeQuery = true)
    List<ProxyLog> findAllAfterId(Long desiredId);
}
