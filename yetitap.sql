-- Create database
CREATE DATABASE IF NOT EXISTS yeticointap;

-- Use the created database
USE yeticointap;

-- Create Users Table
CREATE TABLE Users (
    user_id UUID PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create UserProgress Table
CREATE TABLE UserProgress (
    progress_id UUID PRIMARY KEY,
    user_id UUID,
    level INT DEFAULT 1,
    experience INT DEFAULT 0,
    coins INT DEFAULT 0,
    energy INT DEFAULT 1000,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

-- Create Buildings Table
CREATE TABLE Buildings (
    building_id UUID PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    description TEXT,
    cost INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create UserBuildings Table
CREATE TABLE UserBuildings (
    user_building_id UUID PRIMARY KEY,
    user_id UUID,
    building_id UUID,
    quantity INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (building_id) REFERENCES Buildings(building_id)
);

-- Create Boosts Table
CREATE TABLE Boosts (
    boost_id UUID PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    description TEXT,
    effect VARCHAR(50),
    duration INT NOT NULL,
    cost INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create UserBoosts Table
CREATE TABLE UserBoosts (
    user_boost_id UUID PRIMARY KEY,
    user_id UUID,
    boost_id UUID,
    quantity INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (boost_id) REFERENCES Boosts(boost_id)
);

-- Create Reflinks Table
CREATE TABLE Reflinks (
    reflink_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    reflink_code VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

-- Create UserWallets Table
CREATE TABLE UserWallets (
    wallet_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    address VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);
